"""
ExamForge AI — Flask REST API
Automated Question Paper Generation System for Tamil Nadu State Board (Classes 6–12)
Supervised Random Forest Classification & Role-Based Access Control (Admin vs. Teacher)
"""

import os
import sys
import json
import random
import logging
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_file, g
from flask_cors import CORS

api_logger = logging.getLogger('ExamForgeAPI')

from config import Config
from database.db import SessionLocal, init_db, engine
from database.models import (
    Question, GeneratedPaper, User, Curriculum, DatasetUpload, ModelVersion, AuditLog
)
from ml.predict import predictor
from ml.random_forest_trainer import (
    train_random_forest_pipeline,
    activate_version,
    train_edubloom_candidate_model,
    train_combined_candidate_model
)
from services.edubloom_service import (
    inspect_and_validate_edubloom_dataset,
    optionally_stage_edubloom_into_db,
    build_combined_training_dataset,
    get_combined_dataset_summary
)
from services.question_selection import select_questions
from services.pdf_generator import generate_exam_pdf, generate_teacher_summary_pdf
from services.curriculum_service import (
    get_supported_classes,
    get_supported_subjects,
    get_supported_chapters,
    seed_curriculum_metadata
)
from services.dataset_service import (
    save_uploaded_csv,
    validate_dataset,
    commit_dataset_to_question_bank
)
from services.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    auth_required,
    admin_required,
    teacher_required,
    chairman_required,
    principal_required,
    validate_teacher_subject_access,
    log_audit
)

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": Config.CORS_ALLOWED_ORIGINS}})


def _get_optional_auth_user(db):
    """Retrieve User instance if Authorization Bearer token or ?token= is provided, else None."""
    auth_header = request.headers.get('Authorization')
    token = None
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ', 1)[1].strip()
    elif request.args.get('token'):
        token = request.args.get('token').strip()
        
    if token:
        payload = decode_access_token(token)
        if payload and 'sub' in payload:
            return db.query(User).filter_by(id=int(payload['sub'])).first()
    return None


# --- Initialize database and default accounts on startup ---
def setup_database():
    """Create tables, run migrations, and seed default admin/teacher accounts."""
    init_db()
    db = SessionLocal()
    try:
        # 1. Seed Curriculum metadata
        seed_curriculum_metadata(db)

        # 2. Seed Administrator Account
        admin_email = os.environ.get('DEFAULT_ADMIN_EMAIL', 'admin@examforge.ai')
        admin_pass = os.environ.get('DEFAULT_ADMIN_PASSWORD', 'Admin@2026')
        admin_user = db.query(User).filter_by(email=admin_email).first()
        if not admin_user:
            admin_user = User(
                email=admin_email,
                full_name='System Administrator',
                password_hash=hash_password(admin_pass),
                role='admin',
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            print(f"Created default administrator account: {admin_email}")

        # 3. Seed Teacher Account
        teacher_email = os.environ.get('DEFAULT_TEACHER_EMAIL', 'teacher@examforge.ai')
        teacher_pass = os.environ.get('DEFAULT_TEACHER_PASSWORD', 'ExamForge@2026')
        teacher_user = db.query(User).filter_by(email=teacher_email).first()
        if not teacher_user:
            teacher_user = User(
                email=teacher_email,
                full_name='Lead Faculty / Examiner',
                password_hash=hash_password(teacher_pass),
                role='teacher',
                subject_specialization='Science, Mathematics',
                is_active=True
            )
            db.add(teacher_user)
            db.commit()
            print(f"Created default teacher account: {teacher_email}")
        elif not teacher_user.subject_specialization:
            teacher_user.subject_specialization = 'Science, Mathematics'
            db.commit()

        # 4. Check TN questions count
        tn_count = db.query(Question).filter_by(board='Tamil Nadu State Board').count()
        total_count = db.query(Question).count()
        print(f"Database ready: {tn_count} Tamil Nadu State Board questions (Total questions in DB: {total_count}).")
    finally:
        db.close()


# Initialize database when the app starts (including Gunicorn)
setup_database()

def get_db():
    """SessionLocal generator helper."""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


# ==============================================================================
# SYSTEM & AUTHENTICATION ENDPOINTS
# ==============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with active model status."""
    db = get_db()
    try:
        tn_q_count = db.query(Question).filter_by(board='Tamil Nadu State Board').count()
        model_status = predictor.get_models_status()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'board_supported': 'Tamil Nadu State Board (Classes 6–12)',
            'tn_questions_count': tn_q_count,
            'models': model_status
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        db.close()


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    """Authenticate staff or administrator credentials and return role-scoped JWT token."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request payload must be JSON'}), 400
    
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    db = get_db()
    try:
        user = db.query(User).filter_by(email=email).first()
        if not user or not verify_password(password, user.password_hash):
            return jsonify({'error': 'Invalid email or password. Please verify credentials.'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account has been deactivated. Please contact your system administrator.'}), 403
        
        token = create_access_token(user)
        log_audit(db, action="USER_LOGIN", entity_type="User", entity_id=user.id, user=user)
        
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': user.to_dict()
        }), 200
    finally:
        db.close()


@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    """Register a new faculty member account."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request payload must be JSON'}), 400
    
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    full_name = (data.get('full_name') or '').strip()
    requested_role = (data.get('role') or 'teacher').strip().lower()
    if requested_role in ['chairman', 'principal', 'admin']:
        return jsonify({'error': 'Institutional leadership accounts (Chairman, Principal, Administrator) cannot be created via public registration.'}), 403
    role = 'teacher'
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    if not password or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400
    
    db = get_db()
    try:
        if db.query(User).filter_by(email=email).first():
            return jsonify({'error': 'An account with this email address already exists.'}), 409
        
        new_user = User(
            email=email,
            full_name=full_name or 'Faculty Member',
            password_hash=hash_password(password),
            role=role,
            is_active=True
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        token = create_access_token(new_user)
        log_audit(db, action="USER_REGISTER", entity_type="User", entity_id=new_user.id, user=new_user)
        
        return jsonify({
            'message': 'Account registered successfully',
            'token': token,
            'user': new_user.to_dict()
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({'error': f'Failed to register account: {str(e)}'}), 500
    finally:
        db.close()


@app.route('/api/auth/me', methods=['GET'])
@auth_required
def auth_me():
    """Return profile data of currently authenticated user."""
    return jsonify({
        'user': g.current_user.to_dict()
    })


# ==============================================================================
# DASHBOARD OVERVIEWS (ROLE-SEGREGATED)
# ==============================================================================

@app.route('/api/dashboard/stats', methods=['GET'])
@auth_required
def dashboard_stats():
    """
    Role-aware dashboard statistics:
    - Chairman: Teacher counts (total, active, inactive, grouped by subject) & Principal count.
    - Principal: Teachers count, generation metrics, and subject distributions.
    - Admin: System overview and ML model statuses.
    - Teacher: Papers generated by user, assigned subjects, and verified syllabus stats.
    """
    db = get_db()
    try:
        user = g.current_user
        total_tn_questions = db.query(Question).filter_by(board='Tamil Nadu State Board').count()
        tn_classes = get_supported_classes()
        
        if user.role == 'chairman':
            principal_count = db.query(User).filter(User.role.in_(['principal', 'admin'])).count()
            teacher_query = db.query(User).filter_by(role='teacher')
            total_teachers = teacher_query.count()
            active_teachers = teacher_query.filter_by(is_active=True).count()
            inactive_teachers = teacher_query.filter_by(is_active=False).count()
            
            subject_counts = {}
            unassigned_count = 0
            for t in teacher_query.all():
                assigned = t.assigned_subjects
                if not assigned:
                    unassigned_count += 1
                else:
                    for s in assigned:
                        subject_counts[s] = subject_counts.get(s, 0) + 1
                        
            return jsonify({
                'role': 'chairman',
                'user_name': user.full_name,
                'principals_count': principal_count,
                'teacher_stats': {
                    'total_teachers': total_teachers,
                    'active_teachers': active_teachers,
                    'inactive_teachers': inactive_teachers,
                    'subject_counts': subject_counts,
                    'unassigned_teachers': unassigned_count
                }
            })

        elif user.role == 'principal':
            teacher_query = db.query(User).filter_by(role='teacher')
            total_teachers = teacher_query.count()
            active_teachers = teacher_query.filter_by(is_active=True).count()
            total_papers = db.query(GeneratedPaper).count()
            
            subject_counts = {}
            unassigned_count = 0
            for t in teacher_query.all():
                assigned = t.assigned_subjects
                if not assigned:
                    unassigned_count += 1
                else:
                    for s in assigned:
                        subject_counts[s] = subject_counts.get(s, 0) + 1

            return jsonify({
                'role': 'principal',
                'user_name': user.full_name,
                'teacher_stats': {
                    'total_teachers': total_teachers,
                    'active_teachers': active_teachers,
                    'inactive_teachers': total_teachers - active_teachers,
                    'subject_counts': subject_counts,
                    'unassigned_teachers': unassigned_count
                },
                'papers_generated': total_papers,
                'total_questions_available': total_tn_questions,
                'supported_classes': tn_classes
            })
            
        elif user.role == 'admin':
            staff_count = db.query(User).filter(User.role.in_(['teacher', 'principal'])).count()
            total_papers = db.query(GeneratedPaper).count()
            dataset_uploads_count = db.query(DatasetUpload).count()
            model_versions_count = db.query(ModelVersion).count()
            
            active_bloom = db.query(ModelVersion).filter_by(task_name='bloom_level', is_active=True).first()
            active_diff = db.query(ModelVersion).filter_by(task_name='difficulty', is_active=True).first()
            recent_activity = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(8).all()
            
            return jsonify({
                'role': 'admin',
                'staff_count': staff_count,
                'total_tn_questions': total_tn_questions,
                'papers_generated': total_papers,
                'dataset_uploads_count': dataset_uploads_count,
                'model_versions_count': model_versions_count,
                'active_models': {
                    'bloom': active_bloom.to_dict() if active_bloom else None,
                    'difficulty': active_diff.to_dict() if active_diff else None
                },
                'recent_activity': [a.to_dict() for a in recent_activity]
            })
        else:
            # TEACHER DASHBOARD: Zero model accuracy cards or training summaries!
            teacher_papers = db.query(GeneratedPaper).filter_by(
                created_by_user_id=user.id
            ).order_by(GeneratedPaper.created_at.desc()).limit(6).all()
            
            teacher_papers_count = db.query(GeneratedPaper).filter_by(created_by_user_id=user.id).count()
            
            return jsonify({
                'role': 'teacher',
                'user_name': user.full_name,
                'supported_board': 'Tamil Nadu State Board (Classes 6–12)',
                'supported_classes': tn_classes,
                'total_questions_available': total_tn_questions,
                'papers_generated_by_user': teacher_papers_count,
                'assigned_subjects': user.assigned_subjects,
                'recent_papers': [p.to_dict() for p in teacher_papers]
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


# ==============================================================================
# TAMIL NADU STATE BOARD CURRICULUM ENDPOINTS
# ==============================================================================

@app.route('/api/curriculum/classes', methods=['GET'])
def get_curriculum_classes():
    """Get supported Tamil Nadu State Board classes (Classes 6 to 12)."""
    return jsonify({
        'board': 'Tamil Nadu State Board',
        'classes': get_supported_classes()
    })


@app.route('/api/curriculum/subjects', methods=['GET'])
def get_curriculum_subjects():
    """Get verified subjects for a specific Tamil Nadu State Board class.
    When invoked by an authenticated teacher, returns only their assigned subjects.
    """
    class_grade = request.args.get('class_grade') or 'Class 10'
    all_class_subjects = get_supported_subjects(class_grade)
    
    db = get_db()
    try:
        current_user = _get_optional_auth_user(db)
        if current_user and current_user.role == 'teacher':
            assigned = current_user.assigned_subjects
            if not assigned:
                return jsonify({
                    'board': 'Tamil Nadu State Board',
                    'class_grade': class_grade,
                    'subjects': [],
                    'assigned_subjects': [],
                    'message': 'No subject assigned. Please contact your Principal.'
                })
            assigned_lower = {s.lower(): s for s in assigned}
            allowed = [s for s in all_class_subjects if s.lower() in assigned_lower]
            return jsonify({
                'board': 'Tamil Nadu State Board',
                'class_grade': class_grade,
                'subjects': allowed,
                'assigned_subjects': assigned
            })
        
        return jsonify({
            'board': 'Tamil Nadu State Board',
            'class_grade': class_grade,
            'subjects': all_class_subjects
        })
    finally:
        db.close()


@app.route('/api/curriculum/chapters', methods=['GET'])
def get_curriculum_chapters():
    """Get verified textbook chapters for a specific class and subject.
    When invoked by an authenticated teacher, verifies subject authorization.
    """
    class_grade = request.args.get('class_grade') or 'Class 10'
    subject = request.args.get('subject') or 'Mathematics'
    
    db = get_db()
    try:
        current_user = _get_optional_auth_user(db)
        if current_user and current_user.role == 'teacher':
            is_auth, err_resp, status_code = validate_teacher_subject_access(current_user, subject)
            if not is_auth:
                return jsonify(err_resp), status_code
            
        chapters = get_supported_chapters(class_grade, subject)
        return jsonify({
            'board': 'Tamil Nadu State Board',
            'class_grade': class_grade,
            'subject': subject,
            'chapters': chapters
        })
    finally:
        db.close()


# ==============================================================================
# CHAIRMAN: INSTITUTIONAL PRINCIPAL MANAGEMENT & TEACHER METRICS (CHAIRMAN ONLY)
# ==============================================================================

@app.route('/api/chairman/principals', methods=['GET'])
@chairman_required
def chairman_list_principals():
    """List all registered Principal accounts (Chairman Only)."""
    db = get_db()
    try:
        principals = db.query(User).filter(User.role.in_(['principal', 'admin'])).order_by(User.created_at.desc()).all()
        return jsonify({
            'principals': [p.to_dict() for p in principals],
            'total': len(principals)
        })
    finally:
        db.close()


@app.route('/api/chairman/principals', methods=['POST'])
@chairman_required
def chairman_create_principal():
    """Create a new Principal account (Chairman Only)."""
    data = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()
    full_name = (data.get('full_name') or '').strip()
    phone = (data.get('phone') or '').strip()
    password = data.get('password') or 'Principal@2026'

    if not email:
        return jsonify({'error': 'Principal email is required'}), 400
    if not full_name:
        return jsonify({'error': 'Principal full name is required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400

    db = get_db()
    try:
        if db.query(User).filter_by(email=email).first():
            return jsonify({'error': f'An account with email {email} already exists.'}), 409

        principal = User(
            email=email,
            full_name=full_name,
            phone=phone,
            role='principal',
            password_hash=hash_password(password),
            is_active=True
        )
        db.add(principal)
        db.commit()
        db.refresh(principal)

        log_audit(db, action="CHAIRMAN_CREATE_PRINCIPAL", entity_type="User", entity_id=principal.id, details={'email': email})

        return jsonify({
            'message': f"Principal account for {full_name} created successfully.",
            'principal': principal.to_dict()
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/chairman/principals/<int:user_id>', methods=['PUT'])
@chairman_required
def chairman_update_principal(user_id):
    """Update Principal profile or active status (Chairman Only)."""
    data = request.get_json() or {}
    db = get_db()
    try:
        principal = db.get(User, user_id)
        if not principal:
            return jsonify({'error': 'Principal account not found'}), 404
        if principal.role not in ['principal', 'admin']:
            return jsonify({'error': 'Chairman is strictly authorized to manage Principal accounts only.'}), 400

        if 'full_name' in data and data['full_name']:
            principal.full_name = data['full_name'].strip()
        if 'phone' in data:
            principal.phone = data['phone'].strip()
        if 'is_active' in data:
            principal.is_active = bool(data['is_active'])

        principal.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(principal)

        log_audit(db, action="CHAIRMAN_UPDATE_PRINCIPAL", entity_type="User", entity_id=principal.id, details={'is_active': principal.is_active})

        return jsonify({
            'message': 'Principal account updated successfully.',
            'principal': principal.to_dict()
        })
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/chairman/principals/<int:user_id>/reset-password', methods=['POST'])
@chairman_required
def chairman_reset_principal_password(user_id):
    """Reset password for a Principal account (Chairman Only)."""
    data = request.get_json() or {}
    new_password = data.get('password') or 'Principal@2026'

    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400

    db = get_db()
    try:
        principal = db.get(User, user_id)
        if not principal:
            return jsonify({'error': 'Principal account not found'}), 404
        if principal.role not in ['principal', 'admin']:
            return jsonify({'error': 'Chairman is strictly authorized to manage Principal accounts only.'}), 400

        principal.password_hash = hash_password(new_password)
        principal.updated_at = datetime.now(timezone.utc)
        db.commit()

        log_audit(db, action="CHAIRMAN_RESET_PRINCIPAL_PASSWORD", entity_type="User", entity_id=principal.id)

        return jsonify({
            'message': f"Password for Principal {principal.full_name or principal.email} has been reset successfully."
        })
    finally:
        db.close()


@app.route('/api/chairman/teacher-stats', methods=['GET'])
@chairman_required
def chairman_teacher_stats():
    """
    Chairman dashboard teacher statistics:
    - Total number of teachers
    - Active teachers
    - Inactive teachers
    - Teacher count grouped by subject
    (Counts only - zero password hashes or unnecessary personal info exposed).
    """
    db = get_db()
    try:
        teacher_query = db.query(User).filter_by(role='teacher')
        total_teachers = teacher_query.count()
        active_teachers = teacher_query.filter_by(is_active=True).count()
        inactive_teachers = teacher_query.filter_by(is_active=False).count()

        subject_counts = {}
        unassigned_count = 0
        for t in teacher_query.all():
            assigned = t.assigned_subjects
            if not assigned:
                unassigned_count += 1
            else:
                for s in assigned:
                    subject_counts[s] = subject_counts.get(s, 0) + 1

        return jsonify({
            'total_teachers': total_teachers,
            'active_teachers': active_teachers,
            'inactive_teachers': inactive_teachers,
            'subject_counts': subject_counts,
            'subject_breakdown': subject_counts,
            'unassigned_teachers': unassigned_count
        })
    finally:
        db.close()


# ==============================================================================
# PRINCIPAL: TEACHER MANAGEMENT & SUBJECT ASSIGNMENT (PRINCIPAL / ADMIN)
# ==============================================================================

@app.route('/api/principal/teachers', methods=['GET'])
@principal_required
def principal_list_teachers():
    """List all registered Teacher accounts (Principal / Admin)."""
    db = get_db()
    try:
        teachers = db.query(User).filter_by(role='teacher').order_by(User.created_at.desc()).all()
        return jsonify({
            'teachers': [t.to_dict() for t in teachers],
            'total': len(teachers)
        })
    finally:
        db.close()


@app.route('/api/principal/teachers', methods=['POST'])
@principal_required
def principal_create_teacher():
    """Create a new Teacher account and assign subjects (Principal / Admin)."""
    data = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()
    full_name = (data.get('full_name') or '').strip()
    phone = (data.get('phone') or '').strip()
    password = data.get('password') or 'Teacher@2026'

    raw_subjects = data.get('assigned_subjects') or data.get('subject_specialization')
    if isinstance(raw_subjects, list):
        subject_spec = ', '.join([str(s).strip() for s in raw_subjects if str(s).strip()])
    elif raw_subjects:
        subject_spec = str(raw_subjects).strip()
    else:
        subject_spec = ''

    if not email:
        return jsonify({'error': 'Teacher email is required'}), 400
    if not full_name:
        return jsonify({'error': 'Teacher full name is required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400

    db = get_db()
    try:
        if db.query(User).filter_by(email=email).first():
            return jsonify({'error': f'An account with email {email} already exists.'}), 409

        teacher = User(
            email=email,
            full_name=full_name,
            phone=phone,
            role='teacher',
            subject_specialization=subject_spec,
            password_hash=hash_password(password),
            is_active=True
        )
        db.add(teacher)
        db.commit()
        db.refresh(teacher)

        log_audit(db, action="PRINCIPAL_CREATE_TEACHER", entity_type="User", entity_id=teacher.id, details={'email': email, 'assigned_subjects': teacher.assigned_subjects})

        return jsonify({
            'message': f"Teacher account for {full_name} created successfully.",
            'teacher': teacher.to_dict()
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/principal/teachers/<int:user_id>', methods=['PUT'])
@principal_required
def principal_update_teacher(user_id):
    """Edit Teacher profile, status, and subject assignments (Principal / Admin)."""
    data = request.get_json() or {}
    db = get_db()
    try:
        teacher = db.get(User, user_id)
        if not teacher:
            return jsonify({'error': 'Teacher account not found'}), 404
        if teacher.role != 'teacher':
            return jsonify({'error': 'Can only manage Teacher accounts from this API.'}), 400

        if 'full_name' in data and data['full_name']:
            teacher.full_name = data['full_name'].strip()
        if 'phone' in data:
            teacher.phone = data['phone'].strip()
        if 'is_active' in data:
            teacher.is_active = bool(data['is_active'])

        if 'assigned_subjects' in data or 'subject_specialization' in data:
            raw = data.get('assigned_subjects', data.get('subject_specialization'))
            if isinstance(raw, list):
                teacher.subject_specialization = ', '.join([str(s).strip() for s in raw if str(s).strip()])
            elif raw is not None:
                teacher.subject_specialization = str(raw).strip()

        teacher.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(teacher)

        log_audit(db, action="PRINCIPAL_UPDATE_TEACHER", entity_type="User", entity_id=teacher.id, details={'is_active': teacher.is_active, 'assigned_subjects': teacher.assigned_subjects})

        return jsonify({
            'message': 'Teacher profile updated successfully.',
            'teacher': teacher.to_dict()
        })
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/principal/teachers/<int:user_id>/reset-password', methods=['POST'])
@principal_required
def principal_reset_teacher_password(user_id):
    """Reset password for a Teacher account (Principal / Admin)."""
    data = request.get_json() or {}
    new_password = data.get('password') or 'Teacher@2026'

    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400

    db = get_db()
    try:
        teacher = db.get(User, user_id)
        if not teacher:
            return jsonify({'error': 'Teacher account not found'}), 404
        if teacher.role != 'teacher':
            return jsonify({'error': 'Can only manage Teacher accounts from this API.'}), 400

        teacher.password_hash = hash_password(new_password)
        teacher.updated_at = datetime.now(timezone.utc)
        db.commit()

        log_audit(db, action="PRINCIPAL_RESET_TEACHER_PASSWORD", entity_type="User", entity_id=teacher.id)

        return jsonify({
            'message': f"Password for Teacher {teacher.full_name or teacher.email} has been reset successfully."
        })
    finally:
        db.close()


# ==============================================================================
# ADMIN: STAFF MANAGEMENT ENDPOINTS (ADMIN ONLY)
# ==============================================================================

@app.route('/api/admin/staff', methods=['GET'])
@admin_required
def admin_get_staff():
    """List all registered faculty and staff accounts (Admin Only)."""
    db = get_db()
    try:
        users = db.query(User).order_by(User.created_at.desc()).all()
        return jsonify({
            'staff': [u.to_dict() for u in users],
            'total': len(users)
        })
    finally:
        db.close()


@app.route('/api/admin/staff', methods=['POST'])
@admin_required
def admin_create_staff():
    """Add a new staff or faculty account (Principal / Admin Only)."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request payload must be JSON'}), 400
    
    email = (data.get('email') or '').strip().lower()
    full_name = (data.get('full_name') or '').strip()
    phone = (data.get('phone') or '').strip()
    requested_role = (data.get('role') or 'teacher').strip().lower()
    subject_spec = (data.get('subject_specialization') or '').strip()
    initial_password = data.get('password') or 'Staff@2026'
    
    if not email:
        return jsonify({'error': 'Staff email is required'}), 400
    if not full_name:
        return jsonify({'error': 'Staff full name is required'}), 400
    
    # Strict role hierarchy check
    if requested_role == 'chairman':
        return jsonify({'error': 'Principals/Administrators cannot create Chairman accounts.'}), 403
    if requested_role in ['admin', 'principal']:
        return jsonify({'error': 'Principals cannot create additional Principal/Administrator accounts. Principals must be appointed by the Chairman.'}), 403
    
    role = 'teacher'
    
    db = get_db()
    try:
        existing = db.query(User).filter_by(email=email).first()
        if existing:
            return jsonify({'error': f'An account with email {email} already exists.'}), 409
        
        staff_member = User(
            email=email,
            full_name=full_name,
            phone=phone,
            role=role,
            subject_specialization=subject_spec,
            password_hash=hash_password(initial_password),
            is_active=True
        )
        db.add(staff_member)
        db.commit()
        db.refresh(staff_member)
        
        log_audit(db, action="CREATE_STAFF_ACCOUNT", entity_type="User", entity_id=staff_member.id, details={'email': email, 'role': role})
        
        return jsonify({
            'message': f"Staff account for {full_name} created successfully.",
            'staff': staff_member.to_dict()
        }), 201
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/admin/staff/<int:user_id>', methods=['PUT'])
@admin_required
def admin_update_staff(user_id):
    """Edit staff profile, role, or active status (Admin Only)."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    db = get_db()
    try:
        staff = db.get(User, user_id)
        if not staff:
            return jsonify({'error': 'Staff account not found'}), 404
        
        # Principals/Administrators cannot modify Chairman accounts
        if staff.role == 'chairman':
            return jsonify({'error': 'Principals/Administrators cannot modify Chairman accounts.'}), 403
        
        # Prevent self-deactivation and self-demotion
        if staff.id == g.current_user.id:
            if data.get('is_active') is False:
                return jsonify({'error': 'Cannot deactivate your own administrator account.'}), 400
            if 'role' in data and data['role'] != staff.role:
                return jsonify({'error': 'Cannot change role of your own account (self-demotion forbidden).'}), 400
        
        if 'full_name' in data and data['full_name']:
            staff.full_name = data['full_name'].strip()
        if 'phone' in data:
            staff.phone = data['phone'].strip()
        
        # Role update validations
        if 'role' in data:
            req_role = data['role']
            if req_role == 'chairman':
                return jsonify({'error': 'Cannot promote accounts to Chairman.'}), 403
            if req_role in ['admin', 'principal'] and staff.role not in ['admin', 'principal']:
                return jsonify({'error': 'Principals cannot promote accounts to Principal/Administrator.'}), 403
            if req_role in ['teacher', 'admin']:
                staff.role = req_role
        
        if 'subject_specialization' in data:
            staff.subject_specialization = data['subject_specialization'].strip()
        if 'is_active' in data:
            staff.is_active = bool(data['is_active'])
            
        staff.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(staff)
        
        log_audit(db, action="UPDATE_STAFF_ACCOUNT", entity_type="User", entity_id=staff.id, details={'role': staff.role, 'is_active': staff.is_active})
        
        return jsonify({
            'message': 'Staff details updated successfully',
            'staff': staff.to_dict()
        })
    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/admin/staff/<int:user_id>/reset-password', methods=['POST'])
@admin_required
def admin_reset_staff_password(user_id):
    """Reset password for a staff account (Admin Only)."""
    data = request.get_json() or {}
    new_password = data.get('password') or 'ExamForge@2026'
    
    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400
    
    db = get_db()
    try:
        staff = db.get(User, user_id)
        if not staff:
            return jsonify({'error': 'Staff account not found'}), 404
        if staff.role == 'chairman':
            return jsonify({'error': 'Principals/Administrators cannot reset passwords for Chairman accounts.'}), 403
        
        staff.password_hash = hash_password(new_password)
        staff.updated_at = datetime.now(timezone.utc)
        db.commit()
        
        log_audit(db, action="RESET_STAFF_PASSWORD", entity_type="User", entity_id=staff.id)
        
        return jsonify({
            'message': f"Password for {staff.full_name or staff.email} has been reset successfully."
        })
    finally:
        db.close()


@app.route('/api/admin/staff/<int:user_id>', methods=['DELETE'])
@admin_required
def admin_delete_staff(user_id):
    """Deactivate or remove a staff account (Admin Only)."""
    db = get_db()
    try:
        staff = db.get(User, user_id)
        if not staff:
            return jsonify({'error': 'Staff account not found'}), 404
        if staff.role == 'chairman':
            return jsonify({'error': 'Principals/Administrators cannot delete or deactivate Chairman accounts.'}), 403
        if staff.id == g.current_user.id:
            return jsonify({'error': 'Cannot delete or deactivate your own account'}), 400
        
        staff.is_active = False
        db.commit()
        
        log_audit(db, action="DEACTIVATE_STAFF_ACCOUNT", entity_type="User", entity_id=staff.id)
        return jsonify({'message': f"Staff account {staff.email} deactivated successfully."})
    finally:
        db.close()


# ==============================================================================
# ADMIN: DATASET MANAGEMENT ENDPOINTS (ADMIN ONLY)
# ==============================================================================

@app.route('/api/admin/datasets/upload', methods=['POST'])
@admin_required
def admin_upload_dataset():
    """Upload a CSV dataset containing past Tamil Nadu State Board question papers (Admin Only)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the multipart request'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected for upload'}), 400
    
    result = save_uploaded_csv(file, user_id=g.current_user.id)
    if not result['success']:
        return jsonify(result), 400
    
    return jsonify(result), 201


@app.route('/api/admin/datasets', methods=['GET'])
@admin_required
def admin_list_datasets():
    """List all uploaded datasets with record counts and validation status (Admin Only)."""
    db = get_db()
    try:
        datasets = db.query(DatasetUpload).order_by(DatasetUpload.created_at.desc()).all()
        return jsonify({
            'datasets': [d.to_dict() for d in datasets],
            'total': len(datasets)
        })
    finally:
        db.close()


@app.route('/api/admin/datasets/<int:upload_id>', methods=['GET'])
@admin_required
def admin_get_dataset(upload_id):
    """Get detailed validation report and column mapping for an upload (Admin Only)."""
    db = get_db()
    try:
        upload = db.get(DatasetUpload, upload_id)
        if not upload:
            return jsonify({'error': 'Dataset not found'}), 404
        return jsonify(upload.to_dict())
    finally:
        db.close()


@app.route('/api/admin/datasets/<int:upload_id>/validate', methods=['POST'])
@admin_required
def admin_validate_dataset(upload_id):
    """Validate CSV contents using the mapped columns (Admin Only)."""
    data = request.get_json() or {}
    column_mapping = data.get('column_mapping') or {}
    default_class = data.get('default_class')
    
    result = validate_dataset(
        upload_id=upload_id,
        column_mapping=column_mapping,
        default_class=default_class,
        user_id=g.current_user.id
    )
    if not result['success']:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/admin/datasets/<int:upload_id>/import', methods=['POST'])
@admin_required
def admin_import_dataset_to_bank(upload_id):
    """Ingest validated Tamil Nadu State Board questions into Question Bank (Admin Only)."""
    result = commit_dataset_to_question_bank(upload_id, user_id=g.current_user.id)
    if not result['success']:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/admin/datasets/merge-combined', methods=['POST'])
@admin_required
def admin_merge_combined_dataset():
    """
    Builds or updates the combined training dataset (Existing TN + EduBloom).
    Deduplicates across both sources, normalizes schema and labels,
    and generates manifest with counts and provenance audit (Admin Only).
    """
    data = request.get_json() or {}
    force_rebuild = bool(data.get('force_rebuild', True))
    result = build_combined_training_dataset(force_rebuild=force_rebuild, user_id=g.current_user.id)
    return jsonify(result), 200


@app.route('/api/admin/datasets/combined-summary', methods=['GET'])
@admin_required
def admin_combined_dataset_summary():
    """
    Get manifest and statistics for the combined dataset (Admin Only).
    """
    summary = get_combined_dataset_summary()
    return jsonify(summary), 200


# ==============================================================================
# ADMIN: EDUBLOOM DATASET INTEGRATION & CANDIDATE MODEL PIPELINE (ADMIN ONLY)
# ==============================================================================

@app.route('/api/admin/edubloom/validate', methods=['POST'])
@admin_required
def admin_validate_edubloom():
    """
    Validates the 73,712-row EduBloom dataset (Admin Only).
    Checks columns, missing values, duplicates, and verifies L1-L6 mapping.
    Enforces Tamil Nadu State Board isolation guards.
    """
    report = inspect_and_validate_edubloom_dataset()
    if not report.get('success'):
        return jsonify(report), 400
    return jsonify(report), 200


@app.route('/api/admin/edubloom/train-candidate', methods=['POST'])
@admin_required
def admin_train_edubloom_candidate():
    """Model training through web interface is disabled. Run python backend/ml/train_local.py"""
    return jsonify({
        'success': False,
        'error': 'Model training through web interface is disabled. Please run the local training script in VS Code: python backend/ml/train_local.py'
    }), 403


@app.route('/api/admin/combined/train-candidate', methods=['POST'])
@admin_required
def admin_train_combined_candidate():
    """Model training through web interface is disabled. Run python backend/ml/train_local.py"""
    return jsonify({
        'success': False,
        'error': 'Model training through web interface is disabled. Please run the local training script in VS Code: python backend/ml/train_local.py'
    }), 403



@app.route('/api/admin/edubloom/stage-questions', methods=['POST'])
@admin_required
def admin_stage_edubloom_questions():
    """
    Optionally stages validated EduBloom questions into database (Admin Only).
    STRICT ISOLATION: Tags records as 'General / EduBloom' and verified=False.
    Never marks questions as Tamil Nadu State Board.
    """
    data = request.get_json() or {}
    max_rows = int(data.get('max_rows') or 5000)

    result = optionally_stage_edubloom_into_db(
        max_rows=max_rows,
        user_id=g.current_user.id
    )
    if not result.get('success'):
        return jsonify(result), 400
    return jsonify(result), 200


@app.route('/api/admin/edubloom/summary', methods=['GET'])
@admin_required
def admin_edubloom_summary():
    """
    Summary of EduBloom dataset status, candidate models, and isolation metrics (Admin Only).
    """
    db = get_db()
    try:
        from services.edubloom_service import EDUBLOOM_CSV_PATH, EDUBLOOM_REPORT_PATH
        csv_exists = os.path.exists(EDUBLOOM_CSV_PATH)
        file_size_mb = round(os.path.getsize(EDUBLOOM_CSV_PATH) / (1024 * 1024), 2) if csv_exists else 0

        # Cached validation report
        cached_report = None
        if os.path.exists(EDUBLOOM_REPORT_PATH):
            try:
                with open(EDUBLOOM_REPORT_PATH, 'r', encoding='utf-8') as f:
                    cached_report = json.load(f)
            except Exception:
                cached_report = None

        # Candidate models vs active models
        candidate_models = db.query(ModelVersion).filter(
            ModelVersion.version.like('%candidate%')
        ).order_by(ModelVersion.created_at.desc()).all()

        active_bloom = db.query(ModelVersion).filter_by(task_name='bloom_level', is_active=True).first()
        active_diff = db.query(ModelVersion).filter_by(task_name='difficulty', is_active=True).first()

        # Database counts
        tn_count = db.query(Question).filter(Question.board == 'Tamil Nadu State Board').count()
        general_count = db.query(Question).filter(Question.board != 'Tamil Nadu State Board').count()

        return jsonify({
            'dataset_status': {
                'available': csv_exists,
                'path': EDUBLOOM_CSV_PATH,
                'file_size_mb': file_size_mb,
                'source': 'Hugging Face Uzaib52/edubloom-question-bank'
            },
            'validation_report': cached_report,
            'models': {
                'active_bloom': active_bloom.to_dict() if active_bloom else None,
                'active_difficulty': active_diff.to_dict() if active_diff else None,
                'candidate_models': [m.to_dict() for m in candidate_models]
            },
            'database_isolation': {
                'verified_tn_board_questions': tn_count,
                'general_training_questions': general_count,
                'isolation_enforced': True,
                'policy': 'EduBloom general questions cannot be selected for Tamil Nadu State Board examination papers.'
            }
        })
    finally:
        db.close()


# ==============================================================================
# ADMIN: MODEL TRAINING & EVALUATION ENDPOINTS (ADMIN ONLY)
# ==============================================================================

@app.route('/api/admin/models/train', methods=['POST'])
@admin_required
def admin_train_model():
    """Model training through web interface is disabled. Run python backend/ml/train_local.py"""
    return jsonify({
        'success': False,
        'error': 'Model training through web interface is disabled. Please run the local training script in VS Code: python backend/ml/train_local.py'
    }), 403


@app.route('/api/admin/models/versions', methods=['GET'])
@admin_required
def admin_list_model_versions():
    """List all trained model versions, held-out test metrics, and active states (Admin Only)."""
    db = get_db()
    try:
        task_filter = request.args.get('task')
        query = db.query(ModelVersion)
        if task_filter:
            query = query.filter_by(task_name=task_filter)
        
        versions = query.order_by(ModelVersion.created_at.desc()).all()
        return jsonify({
            'versions': [v.to_dict() for v in versions],
            'total': len(versions)
        })
    finally:
        db.close()


@app.route('/api/admin/models/<int:model_id>/activate', methods=['POST'])
@admin_required
def admin_activate_model(model_id):
    """Activate a validated model version or roll back to a previous version (Admin Only)."""
    result = activate_version(model_id, user_id=g.current_user.id)
    if not result['success']:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/models/evaluation', methods=['GET'])
@admin_required
def models_evaluation():
    """
    Get detailed evaluation metrics for active models and candidate models (ADMIN ONLY).
    Teachers receive 403 Forbidden.
    """
    db = get_db()
    try:
        result = {}
        for task in ['bloom_level', 'difficulty']:
            active_m = db.query(ModelVersion).filter_by(task_name=task, is_active=True).first()
            if active_m and active_m.evaluation_metrics:
                result[task] = active_m.evaluation_metrics
            else:
                eval_file = 'bloom_evaluation.json' if task == 'bloom_level' else 'difficulty_evaluation.json'
                path = os.path.join(Config.REPORTS_DIR, eval_file)
                if os.path.exists(path):
                    with open(path, encoding='utf-8') as f:
                        result[task] = json.load(f)
                else:
                    result[task] = {'status': 'not_trained', 'message': f'Model for {task} has not been trained yet.'}

            # Check for candidate evaluation report
            candidate_file = 'bloom_candidate_evaluation.json' if task == 'bloom_level' else 'difficulty_candidate_evaluation.json'
            cand_path = os.path.join(Config.REPORTS_DIR, candidate_file)
            if os.path.exists(cand_path):
                try:
                    with open(cand_path, encoding='utf-8') as f:
                        result[f"{task}_candidate"] = json.load(f)
                except Exception:
                    pass
        return jsonify(result)
    finally:
        db.close()


@app.route('/api/models/status', methods=['GET'])
@admin_required
def models_status():
    """Get status of trained models (ADMIN ONLY)."""
    status = predictor.get_models_status()
    return jsonify(status)


@app.route('/api/admin/overview', methods=['GET'])
@admin_required
def admin_system_overview():
    """System overview statistics for Administrator."""
    db = get_db()
    try:
        staff_count = db.query(User).filter_by(role='teacher').count()
        total_questions = db.query(Question).filter_by(board='Tamil Nadu State Board').count()
        papers_count = db.query(GeneratedPaper).count()
        datasets_count = db.query(DatasetUpload).count()
        training_runs_count = db.query(ModelVersion).count()
        
        active_models = db.query(ModelVersion).filter_by(is_active=True).all()
        recent_audits = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(10).all()
        
        return jsonify({
            'staff_count': staff_count,
            'total_tamilnadu_questions': total_questions,
            'papers_generated': papers_count,
            'datasets_uploaded': datasets_count,
            'model_training_runs': training_runs_count,
            'active_models': [m.to_dict() for m in active_models],
            'recent_activity': [a.to_dict() for a in recent_audits]
        })
    finally:
        db.close()


# ==============================================================================
# QUESTION BANK & CLASSIFICATION ENDPOINTS
# ==============================================================================

@app.route('/api/questions', methods=['GET'])
@auth_required
def get_questions():
    """Get questions filtered strictly to Tamil Nadu State Board (Classes 6–12) with teacher subject restrictions."""
    db = get_db()
    try:
        user = g.current_user
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        per_page = min(per_page, 100)
        
        # Enforce Tamil Nadu State Board questions
        query = db.query(Question).filter(
            (Question.board == 'Tamil Nadu State Board') | (Question.board.is_(None))
        )
        
        # Teacher subject restriction enforcement
        if user.role == 'teacher':
            assigned = user.assigned_subjects
            if not assigned:
                return jsonify({
                    'questions': [],
                    'total': 0,
                    'page': page,
                    'per_page': per_page,
                    'total_pages': 0,
                    'message': 'No subject assigned. Please contact your Principal.'
                })
            
            subject = request.args.get('subject')
            if subject:
                is_auth, err_resp, status_code = validate_teacher_subject_access(user, subject)
                if not is_auth:
                    return jsonify(err_resp), status_code
                query = query.filter(Question.subject.ilike(subject.strip()))
            else:
                # Case-insensitive match on assigned subjects
                query = query.filter(Question.subject.in_(assigned))
        elif user.role == 'chairman':
            return jsonify({'error': 'Access forbidden. Chairman does not browse question bank.'}), 403
        else:
            subject = request.args.get('subject')
            if subject:
                query = query.filter(Question.subject.ilike(subject.strip()))
            
        # Mandatory/optional filters
        raw_cg = request.args.get('class_grade')
        if raw_cg:
            cg_str = str(raw_cg).strip()
            if cg_str.isdigit():
                cg_str = f"Class {cg_str}"
            query = query.filter(Question.class_grade.ilike(cg_str))
            
        topic = request.args.get('topic') or request.args.get('portion')
        if topic:
            import unicodedata
            nfc_top = unicodedata.normalize('NFC', topic.strip())
            nfd_top = unicodedata.normalize('NFD', topic.strip())
            query = query.filter(Question.topic.in_([nfc_top, nfd_top]))

        marks_param = request.args.get('marks')
        if marks_param:
            try:
                query = query.filter(Question.marks == float(marks_param))
            except (ValueError, TypeError):
                pass
            
        bloom = request.args.get('bloom_level')
        if bloom:
            query = query.filter(Question.bloom_level == bloom)
            
        difficulty = request.args.get('difficulty')
        if difficulty:
            query = query.filter(Question.difficulty == difficulty)
            
        search = request.args.get('search')
        if search:
            query = query.filter(Question.question.contains(search))
            
        total = query.count()
        questions = query.offset((page - 1) * per_page).limit(per_page).all()
        
        return jsonify({
            'questions': [q.to_dict() for q in questions],
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page,
        })
    finally:
        db.close()


@app.route('/api/questions/filters', methods=['GET'])
@auth_required
def get_question_filters():
    """Get available filter options restricted to Tamil Nadu State Board and authorized subjects."""
    db = get_db()
    try:
        user = g.current_user
        classes = get_supported_classes()
        class_param = request.args.get('class_grade')
        
        if class_param:
            subjects = get_supported_subjects(class_param)
        else:
            subjects_q = db.query(Question.subject).filter_by(board='Tamil Nadu State Board').distinct().all()
            subjects = sorted([s[0] for s in subjects_q if s[0]])
            
        if user.role == 'teacher':
            assigned = user.assigned_subjects
            assigned_lower = {s.lower(): s for s in assigned}
            subjects = [s for s in subjects if s.lower() in assigned_lower]
            
        q_types = [t[0] for t in db.query(Question.question_type).distinct().all() if t[0]]
        
        return jsonify({
            'board': 'Tamil Nadu State Board',
            'classes': classes,
            'subjects': subjects,
            'question_types': sorted(q_types) if q_types else ['Short Answer', 'Long Answer', 'Problem Solving', 'Multiple Choice'],
            'bloom_levels': Config.BLOOM_LEVELS,
            'difficulty_levels': Config.DIFFICULTY_LEVELS,
        })
    finally:
        db.close()


@app.route('/api/classify', methods=['POST'])
@auth_required
def classify_question():
    """Classify a question using the active Random Forest models."""
    data = request.get_json()
    q_text = data.get('question') or data.get('text') if data else None
    if not q_text:
        return jsonify({'error': 'Question text is required'}), 400
    
    result = predictor.classify_question(q_text)
    bloom_info = result.get('bloom', {})
    diff_info = result.get('difficulty', {})
    
    return jsonify({
        'question': result.get('question'),
        'bloom_level': bloom_info.get('prediction'),
        'bloom_confidence': bloom_info.get('confidence'),
        'bloom_probabilities': bloom_info.get('probabilities'),
        'difficulty': diff_info.get('prediction'),
        'difficulty_confidence': diff_info.get('confidence'),
        'difficulty_probabilities': diff_info.get('probabilities'),
        'raw': result
    })


# ==============================================================================
# QUESTION PAPER GENERATION WORKFLOW (TEACHERS & ADMINS)
# ==============================================================================

@app.route('/api/generate-paper', methods=['POST'])
@app.route('/api/papers/generate', methods=['POST'])
@teacher_required
def generate_paper():
    """
    Tamil Nadu State Board Question Paper Generation Workflow:
    1. Validates selected class (6 to 12) and subject.
    2. Retrieves approved Tamil Nadu State Board questions.
    3. Respects marks, difficulty, and Bloom taxonomy distributions.
    4. Formats printable PDF with ReportLab.
    5. Saves generation record with user ownership.
    """
    db = get_db()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        # Sanitized request logging (strictly excluding authorization tokens, passwords, cookies)
        sanitized_summary = {
            'class_grade': data.get('class_grade'),
            'subject': data.get('subject'),
            'board': data.get('board'),
            'chapters': data.get('chapters') or data.get('portions') or data.get('chapter'),
            'sections': data.get('sections') or data.get('sections_config'),
            'num_questions': data.get('num_questions'),
            'total_marks': data.get('total_marks')
        }
        app.logger.info(f"[GENERATE_PAPER_REQUEST] {json.dumps(sanitized_summary, ensure_ascii=False)}")
        api_logger.info(f"[GENERATE_PAPER_REQUEST] {json.dumps(sanitized_summary, ensure_ascii=False)}")
        
        title = (data.get('title') or 'Terminal Examination').strip()
        raw_class = data.get('class_grade')
        if raw_class:
            class_str = str(raw_class).strip()
            if class_str.isdigit():
                class_grade = f"Class {class_str}"
            elif class_str.lower().startswith('class'):
                parts = class_str.split()
                if len(parts) == 2 and parts[1].isdigit():
                    class_grade = f"Class {parts[1]}"
                else:
                    class_grade = class_str
            else:
                class_grade = class_str
        else:
            class_grade = None

        # Board Validation
        raw_board = data.get('board')
        from services.question_selection import normalize_board
        if raw_board and normalize_board(raw_board) != 'Tamil Nadu State Board':
            return jsonify({
                'success': False,
                'is_valid': False,
                'error': f"Unsupported educational board '{raw_board}'. The application strictly supports Tamil Nadu State Board / Samacheer Kalvi."
            }), 400

        # Mandatory Class Validation: Classes 6 to 12 only!
        if not class_grade or class_grade not in get_supported_classes():
            return jsonify({
                'success': False,
                'error': f"Class selection is mandatory and must be one of Classes 6 to 12. Received: '{raw_class}'."
            }), 400

        # Mandatory Subject Validation
        raw_subject = data.get('subject')
        valid_subjects = get_supported_subjects(class_grade)
        subject = None
        if raw_subject:
            s_str = str(raw_subject).strip()
            for vs in valid_subjects:
                if vs.lower() == s_str.lower():
                    subject = vs
                    break
            if not subject:
                subject = s_str

        if not subject:
            return jsonify({
                'success': False,
                'is_valid': False,
                'error': f"Subject selection is mandatory for {class_grade}."
            }), 400

        # Teacher Subject Authorization Check
        is_auth, err_resp, status_code = validate_teacher_subject_access(g.current_user, subject)
        if not is_auth:
            return jsonify(err_resp), status_code

        # Unicode normalization (NFC & NFD) for portions/chapters
        import unicodedata
        raw_chapters = data.get('chapters') or data.get('portions') or data.get('chapter') or []
        if isinstance(raw_chapters, str):
            raw_chapters = [c.strip() for c in raw_chapters.split(',') if c.strip()]

        chapters = []
        nfc_nfd_chaps = set()
        for c in raw_chapters:
            if c:
                norm_c = unicodedata.normalize('NFC', str(c).strip())
                if norm_c and norm_c not in chapters:
                    chapters.append(norm_c)
                    nfc_nfd_chaps.add(norm_c)
                    nfc_nfd_chaps.add(unicodedata.normalize('NFD', norm_c))

        raw_topic = data.get('topic') or data.get('portion')
        topic = unicodedata.normalize('NFC', str(raw_topic).strip()) if raw_topic else None
        if topic and topic.lower() not in ['complete syllabus', 'all chapters', 'all']:
            if topic not in chapters:
                chapters.append(topic)
            nfc_nfd_chaps.add(topic)
            nfc_nfd_chaps.add(unicodedata.normalize('NFD', topic))

        duration_minutes = int(data.get('duration_minutes') or 90)
        num_questions = int(data.get('num_questions') or 10)
        total_marks = float(data.get('total_marks') or (num_questions * 2))

        # Compute true syllabus portions display (never truncate silently)
        all_subject_chapters = get_supported_chapters(class_grade, subject)
        if chapters and isinstance(chapters, list) and len(chapters) > 0:
            if len(chapters) == len(all_subject_chapters) and len(all_subject_chapters) > 0:
                topic_display = "Complete Syllabus (All Chapters)"
            else:
                topic_display = ", ".join(chapters)
        elif topic:
            topic_display = topic
        else:
            topic_display = "Complete Syllabus"

        # Query approved Tamil Nadu State Board questions
        query = db.query(Question).filter(
            (Question.board == 'Tamil Nadu State Board') |
            (Question.board == 'Tamil NaduState Board') |
            (Question.board.ilike('Tamil Nadu%')) |
            (Question.board.is_(None))
        ).filter(
            Question.class_grade.ilike(class_grade)
        ).filter(
            Question.subject.ilike(subject)
        )

        # If specific chapters/topics are chosen
        if nfc_nfd_chaps:
            query = query.filter(Question.topic.in_(list(nfc_nfd_chaps)))
            
        eligible_questions = [q.to_dict() for q in query.all()]
        
        if not eligible_questions:
            return jsonify({
                'success': False,
                'is_valid': False,
                'error': f"No questions found in the question bank for Tamil Nadu State Board {class_grade} {subject} with the selected chapters.",
                'eligible_count': 0,
                'suggestions': [
                    "Select all chapters or a broader chapter set",
                    "Verify questions have been ingested for this subject in the Question Bank",
                    "Upload past papers via the Admin Dataset panel if you are an administrator"
                ]
            }), 400
        
        # Difficulty & Bloom distributions
        diff_dist = data.get('difficulty_distribution')
        bloom_dist = data.get('bloom_distribution')
        sections_config = data.get('sections') or data.get('sections_config')
        seed = data.get('seed') or random.randint(1000, 999999)
        
        # Select questions using intelligent selection service
        result = select_questions(
            available_questions=eligible_questions,
            num_questions=num_questions,
            total_marks=total_marks,
            subject=subject,
            topic=topic,
            topics=chapters,
            board='Tamil Nadu State Board',
            class_grade=class_grade,
            difficulty_distribution=diff_dist,
            bloom_distribution=bloom_dist,
            duration_minutes=duration_minutes,
            sections_config=sections_config,
            seed=seed,
            language=data.get('language')
        )
        
        if not result['success']:
            return jsonify(result), 400
        
        selected_questions = result['selected_questions']
        sections = result.get('sections', [])
        
        # Generate official printable PDF with accurate portions and integer marks
        paper_meta = {
            'title': title,
            'board': 'Tamil Nadu State Board',
            'class_grade': class_grade,
            'subject': subject,
            'topic': topic_display,
            'duration_minutes': duration_minutes,
            'total_marks': result['total_marks'],
        }
        
        filepath, filename = generate_exam_pdf(
            paper_data=paper_meta,
            questions=selected_questions,
            sections=sections
        )
        
        # Persist generated paper with user ownership
        paper = GeneratedPaper(
            title=title,
            board='Tamil Nadu State Board',
            class_grade=class_grade,
            subject=subject,
            topic=topic_display,
            total_marks=result['total_marks'],
            duration_minutes=duration_minutes,
            num_questions=len(selected_questions),
            question_type=data.get('question_type', 'Mixed'),
            bloom_level=data.get('bloom_level'),
            difficulty=data.get('difficulty'),
            questions_json=selected_questions,
            sections_json=sections,
            pdf_filename=filename,
            generation_seed=seed,
            created_by_user_id=g.current_user.id
        )
        db.add(paper)
        db.commit()
        db.refresh(paper)
        
        log_audit(db, action="GENERATE_QUESTION_PAPER", entity_type="GeneratedPaper", entity_id=paper.id)
        
        return jsonify({
            'success': True,
            'is_valid': result.get('is_valid', True),
            'paper_id': paper.id,
            'title': title,
            'class_grade': class_grade,
            'subject': subject,
            'portions': topic_display,
            'duration_minutes': duration_minutes,
            'questions': selected_questions,
            'sections': sections,
            'count': len(selected_questions),
            'total_marks': result['total_marks'],
            'pdf_filename': filename,
            'bloom_distribution': result['bloom_distribution'],
            'difficulty_distribution': result['difficulty_distribution'],
            'validation_checks': result.get('validation_checks', {}),
            'warnings': result['warnings'],
        }), 201

    except Exception as e:
        db.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/generate-paper/preview', methods=['POST'])
@app.route('/api/generate-paper/validate', methods=['POST'])
@teacher_required
def preview_paper_blueprint():
    """
    Teacher Blueprint Preview & Pre-Export Validation:
    Allows teachers to inspect syllabus coverage, marks distribution, section totals,
    continuous numbering, and quality warnings BEFORE committing or exporting the PDF.
    """
    db = get_db()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body is required'}), 400

        # Sanitized request logging (strictly excluding authorization tokens, passwords, cookies)
        sanitized_summary = {
            'class_grade': data.get('class_grade'),
            'subject': data.get('subject'),
            'board': data.get('board'),
            'chapters': data.get('chapters') or data.get('portions') or data.get('chapter'),
            'sections': data.get('sections') or data.get('sections_config'),
            'num_questions': data.get('num_questions'),
            'total_marks': data.get('total_marks')
        }
        app.logger.info(f"[GENERATE_PAPER_REQUEST] {json.dumps(sanitized_summary, ensure_ascii=False)}")
        api_logger.info(f"[GENERATE_PAPER_REQUEST] {json.dumps(sanitized_summary, ensure_ascii=False)}")

        title = (data.get('title') or 'Terminal Examination').strip()
        raw_class = data.get('class_grade')
        if raw_class:
            class_str = str(raw_class).strip()
            if class_str.isdigit():
                class_grade = f"Class {class_str}"
            elif class_str.lower().startswith('class'):
                parts = class_str.split()
                if len(parts) == 2 and parts[1].isdigit():
                    class_grade = f"Class {parts[1]}"
                else:
                    class_grade = class_str
            else:
                class_grade = class_str
        else:
            class_grade = None

        # Board Validation
        raw_board = data.get('board')
        from services.question_selection import normalize_board
        if raw_board and normalize_board(raw_board) != 'Tamil Nadu State Board':
            return jsonify({
                'success': False,
                'is_valid': False,
                'error': f"Unsupported educational board '{raw_board}'. The application strictly supports Tamil Nadu State Board / Samacheer Kalvi."
            }), 400

        # Mandatory Class Validation: Classes 6 to 12 only!
        if not class_grade or class_grade not in get_supported_classes():
            return jsonify({
                'success': False,
                'is_valid': False,
                'error': f"Class selection is mandatory and must be one of Classes 6 to 12. Received: '{raw_class}'."
            }), 400

        # Mandatory Subject Validation & Teacher Permission Check
        raw_subject = data.get('subject')
        valid_subjects = get_supported_subjects(class_grade)
        subject = None
        if raw_subject:
            s_str = str(raw_subject).strip()
            for vs in valid_subjects:
                if vs.lower() == s_str.lower():
                    subject = vs
                    break
            if not subject:
                subject = s_str

        if not subject:
            return jsonify({
                'success': False,
                'is_valid': False,
                'error': f"Subject selection is mandatory for {class_grade}."
            }), 400

        is_auth, err_resp, status_code = validate_teacher_subject_access(g.current_user, subject)
        if not is_auth:
            return jsonify(err_resp), status_code

        # Unicode normalization (NFC & NFD) for portions/chapters
        import unicodedata
        raw_chapters = data.get('chapters') or data.get('portions') or data.get('chapter') or []
        if isinstance(raw_chapters, str):
            raw_chapters = [c.strip() for c in raw_chapters.split(',') if c.strip()]

        chapters = []
        nfc_nfd_chaps = set()
        for c in raw_chapters:
            if c:
                norm_c = unicodedata.normalize('NFC', str(c).strip())
                if norm_c and norm_c not in chapters:
                    chapters.append(norm_c)
                    nfc_nfd_chaps.add(norm_c)
                    nfc_nfd_chaps.add(unicodedata.normalize('NFD', norm_c))

        raw_topic = data.get('topic') or data.get('portion')
        topic = unicodedata.normalize('NFC', str(raw_topic).strip()) if raw_topic else None
        if topic and topic.lower() not in ['complete syllabus', 'all chapters', 'all']:
            if topic not in chapters:
                chapters.append(topic)
            nfc_nfd_chaps.add(topic)
            nfc_nfd_chaps.add(unicodedata.normalize('NFD', topic))

        duration_minutes = int(data.get('duration_minutes') or 90)
        num_questions = int(data.get('num_questions') or 10)
        total_marks = float(data.get('total_marks') or (num_questions * 2))

        # Compute true syllabus portions display
        all_subject_chapters = get_supported_chapters(class_grade, subject)
        if chapters and isinstance(chapters, list) and len(chapters) > 0:
            if len(chapters) == len(all_subject_chapters) and len(all_subject_chapters) > 0:
                topic_display = "Complete Syllabus (All Chapters)"
            else:
                topic_display = ", ".join(chapters)
        elif topic:
            topic_display = topic
        else:
            topic_display = "Complete Syllabus"

        # Query approved Tamil Nadu State Board questions
        query = db.query(Question).filter(
            (Question.board == 'Tamil Nadu State Board') |
            (Question.board == 'Tamil NaduState Board') |
            (Question.board.ilike('Tamil Nadu%')) |
            (Question.board.is_(None))
        ).filter(
            Question.class_grade.ilike(class_grade)
        ).filter(
            Question.subject.ilike(subject)
        )

        if nfc_nfd_chaps:
            query = query.filter(Question.topic.in_(list(nfc_nfd_chaps)))

        eligible_questions = [q.to_dict() for q in query.all()]

        if not eligible_questions:
            return jsonify({
                'success': False,
                'is_valid': False,
                'error': f"No questions found in question bank for Tamil Nadu State Board {class_grade} {subject} matching the selected syllabus portions: '{topic_display}'.",
                'eligible_count': 0,
                'suggestions': [
                    "Select additional chapters from the syllabus",
                    "Verify questions have been ingested for this subject in the Question Bank"
                ]
            }), 400

        diff_dist = data.get('difficulty_distribution')
        bloom_dist = data.get('bloom_distribution')
        sections_config = data.get('sections') or data.get('sections_config')
        seed = data.get('seed') or random.randint(1000, 999999)

        result = select_questions(
            available_questions=eligible_questions,
            num_questions=num_questions,
            total_marks=total_marks,
            subject=subject,
            topic=topic,
            topics=chapters,
            board='Tamil Nadu State Board',
            class_grade=class_grade,
            difficulty_distribution=diff_dist,
            bloom_distribution=bloom_dist,
            duration_minutes=duration_minutes,
            sections_config=sections_config,
            seed=seed,
            language=data.get('language')
        )

        if not result['success']:
            return jsonify(result), 400

        dur_str = f"{duration_minutes // 60}h {duration_minutes % 60}m" if duration_minutes >= 60 else f"{duration_minutes} Mins"

        return jsonify({
            'success': True,
            'is_valid': result.get('is_valid', True),
            'title': title,
            'class_grade': class_grade,
            'subject': subject,
            'portions': topic_display,
            'duration_minutes': duration_minutes,
            'duration_display': dur_str,
            'num_questions': result['count'],
            'total_marks': result['total_marks'],
            'sections': result.get('sections', []),
            'questions': result.get('selected_questions', []),
            'bloom_distribution': result.get('bloom_distribution', {}),
            'difficulty_distribution': result.get('difficulty_distribution', {}),
            'validation_checks': result.get('validation_checks', {}),
            'warnings': result.get('warnings', []),
            'seed': seed
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/history', methods=['GET'])
@auth_required
def generation_history():
    """
    Get question paper history.
    Teachers view ONLY their own papers for their assigned subjects.
    Admins and Principals can view all papers.
    """
    db = get_db()
    try:
        user = g.current_user
        query = db.query(GeneratedPaper)
        
        if user.role == 'teacher':
            assigned = user.assigned_subjects
            query = query.filter_by(created_by_user_id=user.id)
            if assigned:
                query = query.filter(GeneratedPaper.subject.in_(assigned))
            else:
                return jsonify({'papers': [], 'total': 0})
        elif user.role == 'chairman':
            return jsonify({'error': 'Access forbidden. Chairman does not manage question papers.'}), 403
            
        papers = query.order_by(GeneratedPaper.created_at.desc()).all()
        return jsonify({
            'papers': [p.to_dict() for p in papers],
            'total': len(papers)
        })
    finally:
        db.close()


@app.route('/api/papers/<int:paper_id>', methods=['GET'])
@auth_required
def get_paper(paper_id):
    """Get details of a specific generated question paper with subject & ownership check."""
    db = get_db()
    try:
        paper = db.get(GeneratedPaper, paper_id)
        if not paper:
            return jsonify({'error': 'Question paper not found'}), 404
        
        # Check authorization: user can access if admin/principal or paper creator with assigned subject
        if g.current_user.role not in ['admin', 'principal']:
            if paper.created_by_user_id and paper.created_by_user_id != g.current_user.id:
                return jsonify({'error': 'Unauthorized to access this examination paper.'}), 403
            is_auth, err_resp, status_code = validate_teacher_subject_access(g.current_user, paper.subject)
            if not is_auth:
                return jsonify(err_resp), status_code
            
        return jsonify(paper.to_dict())
    finally:
        db.close()


@app.route('/api/papers/<int:paper_id>/download', methods=['GET'])
def download_paper(paper_id):
    """Download a generated examination paper PDF with ownership and subject authorization."""
    db = get_db()
    try:
        paper = db.get(GeneratedPaper, paper_id)
        if not paper or not paper.pdf_filename:
            return jsonify({'error': 'Paper or PDF file not found'}), 404
        
        current_user = _get_optional_auth_user(db)
        if current_user and current_user.role not in ['admin', 'principal']:
            if paper.created_by_user_id and paper.created_by_user_id != current_user.id:
                return jsonify({'error': 'Unauthorized to download this examination paper.'}), 403
            is_auth, err_resp, status_code = validate_teacher_subject_access(current_user, paper.subject)
            if not is_auth:
                return jsonify(err_resp), status_code
        
        filepath = os.path.join(Config.GENERATED_PAPERS_DIR, paper.pdf_filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'PDF file not found on disk'}), 404
        
        paper.status = 'downloaded'
        db.commit()
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=paper.pdf_filename,
            mimetype='application/pdf'
        )
    finally:
        db.close()


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == '__main__':
    print("╔" + "═" * 68 + "╗")
    print("║  ExamForge AI — Tamil Nadu State Board (Classes 6–12) Server       ║")
    print("╚" + "═" * 68 + "╝")
    
    setup_database()
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
