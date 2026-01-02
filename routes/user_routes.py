# routes/user_routes.py - User management and attendance routes
from flask import Blueprint, jsonify, current_app
import traceback

# Create blueprint for user routes
bp = Blueprint('user', __name__, url_prefix='/api')

@bp.route('/users', methods=['GET'])
def list_users():
    """Get list of all registered users"""
    try:
        app = current_app
        db = app.config.get('db')
        
        if db is None:
            return jsonify({
                'success': False,
                'error': 'Database not available',
                'users': []
            }), 503
        
        if hasattr(db, 'get_all_users'):
            users = db.get_all_users()
            return jsonify({
                'success': True,
                'users': users,
                'count': len(users)
            })
        else:
            return jsonify({
                'success': True,
                'users': [],
                'message': 'get_all_users method not available',
                'count': 0
            })
        
    except Exception as e:
        print(f"❌ Users list error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'users': []
        }), 500

@bp.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get specific user by ID"""
    try:
        app = current_app
        db = app.config.get('db')
        
        if db is None:
            return jsonify({
                'success': False,
                'error': 'Database not available',
                'user': None
            }), 503
        
        # Try different methods to get user
        if hasattr(db, 'get_user_by_id'):
            user = db.get_user_by_id(user_id)
        elif hasattr(db, 'find_user_by_id'):
            user = db.find_user_by_id(user_id)
        elif hasattr(db, 'get_all_users'):
            # Fallback: filter from all users
            users = db.get_all_users()
            user = next((u for u in users if u.get('user_id') == user_id or u.get('id') == user_id), None)
        else:
            user = None
        
        if user:
            return jsonify({
                'success': True,
                'user': user
            })
        else:
            return jsonify({
                'success': False,
                'error': f'User {user_id} not found',
                'user': None
            }), 404
            
    except Exception as e:
        print(f"❌ Get user error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'user': None
        }), 500

@bp.route('/attendance/<student_id>', methods=['GET'])
def get_attendance(student_id):
    """Get attendance records for a student"""
    try:
        app = current_app
        db = app.config.get('db')
        
        print(f"📊 Fetching attendance for student: {student_id}")
        
        if db is None:
            print("⚠️ Database not available, returning mock data")
            # Return mock attendance data
            mock_data = [
                {
                    'student_id': student_id,
                    'attendance_date': '2024-01-15',
                    'attendance_time': '09:30:00',
                    'attendance_status': 'present',
                    'confidence_score': 0.95,
                    'attendance_type': 'class'
                },
                {
                    'student_id': student_id,
                    'attendance_date': '2024-01-16',
                    'attendance_time': '09:45:00',
                    'attendance_status': 'late',
                    'confidence_score': 0.87,
                    'attendance_type': 'class'
                },
                {
                    'student_id': student_id,
                    'attendance_date': '2024-01-17',
                    'attendance_time': '09:15:00',
                    'attendance_status': 'present',
                    'confidence_score': 0.91,
                    'attendance_type': 'class'
                }
            ]
            return jsonify({
                'success': True,
                'message': 'Using mock data (database not available)',
                'records': mock_data,
                'count': len(mock_data)
            })
        
        # Check if the method exists in database manager
        if hasattr(db, 'get_attendance_records'):
            records = db.get_attendance_records(student_id)
            print(f"✅ Found {len(records)} attendance records")
            return jsonify({
                'success': True,
                'records': records,
                'count': len(records)
            })
        elif hasattr(db, 'get_student_attendance'):
            records = db.get_student_attendance(student_id)
            print(f"✅ Found {len(records)} attendance records")
            return jsonify({
                'success': True,
                'records': records,
                'count': len(records)
            })
        else:
            print("⚠️ No attendance method found in database manager")
            # Return empty records if method doesn't exist
            return jsonify({
                'success': True,
                'message': 'Attendance method not implemented in database',
                'records': [],
                'count': 0
            })
            
    except Exception as e:
        print(f"❌ Attendance fetch error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'records': []
        }), 500

@bp.route('/attendance/mark', methods=['POST'])
def mark_attendance():
    """Mark attendance for a student"""
    try:
        app = current_app
        db = app.config.get('db')
        
        if db is None:
            return jsonify({
                'success': False,
                'error': 'Database not available',
                'attendance_id': None
            }), 503
        
        # Get request data
        data = current_app.request.json
        
        if not data or 'student_id' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing student_id in request',
                'attendance_id': None
            }), 400
        
        student_id = data['student_id']
        attendance_date = data.get('attendance_date')  # Optional, defaults to today
        attendance_time = data.get('attendance_time')  # Optional, defaults to now
        attendance_status = data.get('attendance_status', 'present')
        confidence_score = data.get('confidence_score', 0.0)
        attendance_type = data.get('attendance_type', 'class')
        
        # Check if method exists to mark attendance
        if hasattr(db, 'mark_attendance'):
            attendance_id = db.mark_attendance(
                student_id=student_id,
                attendance_date=attendance_date,
                attendance_time=attendance_time,
                attendance_status=attendance_status,
                confidence_score=confidence_score,
                attendance_type=attendance_type
            )
            
            if attendance_id:
                return jsonify({
                    'success': True,
                    'message': 'Attendance marked successfully',
                    'attendance_id': attendance_id,
                    'student_id': student_id
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to mark attendance',
                    'attendance_id': None
                }), 500
        elif hasattr(db, 'record_attendance'):
            success = db.record_attendance(
                student_id=student_id,
                status=attendance_status,
                confidence=confidence_score
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Attendance recorded successfully',
                    'student_id': student_id
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to record attendance',
                    'attendance_id': None
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Attendance marking not implemented in database',
                'attendance_id': None
            }), 501
            
    except Exception as e:
        print(f"❌ Mark attendance error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'attendance_id': None
        }), 500

@bp.route('/users/delete/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user from the system"""
    try:
        app = current_app
        db = app.config.get('db')
        
        if db is None:
            return jsonify({
                'success': False,
                'error': 'Database not available'
            }), 503
        
        # Check if method exists
        if hasattr(db, 'delete_user'):
            success = db.delete_user(user_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'User {user_id} deleted successfully'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Failed to delete user {user_id}'
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'User deletion not implemented in database',
                'message': f'User {user_id} would be deleted if implemented'
            }), 501
            
    except Exception as e:
        print(f"❌ Delete user error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@bp.route('/attendance/summary/<student_id>', methods=['GET'])
def get_attendance_summary(student_id):
    """Get attendance summary for a student"""
    try:
        app = current_app
        db = app.config.get('db')
        
        if db is None:
            # Return mock summary
            return jsonify({
                'success': True,
                'message': 'Using mock summary (database not available)',
                'student_id': student_id,
                'summary': {
                    'total_days': 30,
                    'present_days': 25,
                    'absent_days': 3,
                    'late_days': 2,
                    'attendance_rate': 83.33,
                    'current_streak': 5
                }
            })
        
        # Check if method exists
        if hasattr(db, 'get_attendance_summary'):
            summary = db.get_attendance_summary(student_id)
            return jsonify({
                'success': True,
                'student_id': student_id,
                'summary': summary
            })
        else:
            # Calculate summary from records
            records = []
            if hasattr(db, 'get_attendance_records'):
                records = db.get_attendance_records(student_id)
            elif hasattr(db, 'get_student_attendance'):
                records = db.get_student_attendance(student_id)
            
            if records:
                total_days = len(records)
                present_days = len([r for r in records if r.get('attendance_status') in ['present', 'late']])
                absent_days = len([r for r in records if r.get('attendance_status') == 'absent'])
                late_days = len([r for r in records if r.get('attendance_status') == 'late'])
                attendance_rate = (present_days / total_days * 100) if total_days > 0 else 0
                
                summary = {
                    'total_days': total_days,
                    'present_days': present_days,
                    'absent_days': absent_days,
                    'late_days': late_days,
                    'attendance_rate': round(attendance_rate, 2),
                    'current_streak': 0  # Would need date calculation
                }
                
                return jsonify({
                    'success': True,
                    'student_id': student_id,
                    'summary': summary
                })
            else:
                return jsonify({
                    'success': True,
                    'student_id': student_id,
                    'summary': {
                        'total_days': 0,
                        'present_days': 0,
                        'absent_days': 0,
                        'late_days': 0,
                        'attendance_rate': 0.0,
                        'current_streak': 0
                    },
                    'message': 'No attendance records found'
                })
            
    except Exception as e:
        print(f"❌ Attendance summary error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'student_id': student_id,
            'summary': None
        }), 500