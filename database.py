# database.py - Updated version
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from datetime import datetime
import json
from typing import Optional, List, Dict, Any

class DatabaseManager:
    def __init__(self, host="localhost", database="attendance_db", 
                 user="postgres", password="root", port=5432):
        """Initialize database connection"""
        self.connection_params = {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
            "port": port
        }
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.conn.autocommit = True
            print("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            print(f"   Host: {self.connection_params['host']}")
            print(f"   Database: {self.connection_params['database']}")
            print(f"   User: {self.connection_params['user']}")
            self.conn = None
            return False
    
    def is_connected(self) -> bool:
        """Check if database connection is active"""
        if self.conn is None:
            return False
    
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("🔌 Database connection closed")
    
    def save_face_template(self, person_id: str, person_name: str, 
                          embedding: np.ndarray, metadata: Optional[dict] = None) -> Optional[int]:
        """Save face template to database"""
        if not self.is_connected():
            print("❌ Database connection is not active")
            return None
    
        cursor = None
        try:
            cursor = self.conn.cursor()
        
            # Convert numpy array to list for PostgreSQL
            embedding_list = embedding.tolist()
            embedding_dim = len(embedding_list)
        
            # Prepare metadata - handle None case
            if metadata is None:
                metadata_json = json.dumps({})  # Default to empty dict
            else:
                metadata_json = json.dumps(metadata)
        
            # Insert face template
            cursor.execute("""
                INSERT INTO face_templates (person_id, person_name, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (person_id, person_name, embedding_list, metadata_json))
        
            face_template_id = cursor.fetchone()[0]
            print(f"✅ Face template saved with ID: {face_template_id}")
            print(f"   Embedding dimension: {embedding_dim}")
        
            return face_template_id
        
        except Exception as e:
            print(f"❌ Error saving face template: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def register_user(self, user_data: Dict[str, Any], face_templates_data: List[Dict[str, Any]]) -> bool:
        """Register a new user with face templates"""
        if not self.is_connected():
            return False
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            # Start transaction
            cursor.execute("BEGIN")
            
            # 1. Save face templates first
            face_template_ids = []
            for template in face_templates_data:
                template_id = self.save_face_template(
                    template['person_id'],
                    template['person_name'],
                    np.array(template['embedding']),
                    template.get('metadata', {})
                )
                if template_id:
                    face_template_ids.append(template_id)
            
            if not face_template_ids:
                cursor.execute("ROLLBACK")
                print("❌ No face templates saved")
                return False
            
            # 2. Save user (without additional_info field as per schema)
            cursor.execute("""
                INSERT INTO users (
                    user_id, person_id, full_name, user_type, 
                    email, phone, date_of_birth, department
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
            """, (
                user_data['user_id'],
                user_data['person_id'],
                user_data['full_name'],
                user_data['user_type'],
                user_data.get('email'),
                user_data.get('phone'),
                user_data.get('date_of_birth'),
                user_data.get('department')
            ))
            
            # 3. Save role-specific data
            user_type = user_data['user_type']
            if user_type == 'student' and 'student' in user_data:
                student_data = user_data['student']
                cursor.execute("""
                    INSERT INTO students (
                        student_id, user_id, full_name, enrollment_number,
                        semester, program, batch_year, email, phone, date_of_birth
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (student_id) DO NOTHING
                """, (
                    student_data['student_id'],
                    student_data['user_id'],
                    student_data['full_name'],
                    student_data.get('enrollment_number'),
                    student_data.get('semester'),
                    student_data.get('program'),
                    student_data.get('batch_year'),
                    student_data.get('email'),
                    student_data.get('phone'),
                    student_data.get('date_of_birth')
                ))
            
            elif user_type == 'faculty' and 'faculty' in user_data:
                faculty_data = user_data['faculty']
                cursor.execute("""
                    INSERT INTO faculty (
                        faculty_id, user_id, designation, qualification
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (faculty_id) DO NOTHING
                """, (
                    faculty_data['faculty_id'],
                    faculty_data['user_id'],
                    faculty_data.get('designation'),
                    faculty_data.get('qualification')
                ))
            
            cursor.execute("COMMIT")
            print(f"✅ User {user_data['full_name']} registered successfully")
            return True
            
        except Exception as e:
            if cursor:
                cursor.execute("ROLLBACK")
            print(f"❌ Error registering user: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def record_student_attendance(self, attendance_data: Dict[str, Any]) -> Optional[int]:
        """Record student attendance in database"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            # Get current timestamp for attendance_time
            attendance_time = attendance_data.get('attendance_time', datetime.now())
            
            cursor.execute("""
                INSERT INTO student_attendance (
                    student_id, user_id, person_id,
                    attendance_date, attendance_time, attendance_type,
                    attendance_status, confidence_score, face_template_id,
                    authentication_method
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING attendance_id
            """, (
                attendance_data['student_id'],
                attendance_data['user_id'],
                attendance_data['person_id'],
                attendance_time.date(),
                attendance_time,
                attendance_data.get('attendance_type', 'class'),
                attendance_data.get('attendance_status', 'present'),
                attendance_data.get('confidence_score'),
                attendance_data.get('face_template_id'),
                attendance_data.get('authentication_method', 'face')
            ))
            
            attendance_id = cursor.fetchone()[0]
            print(f"✅ Student attendance recorded with ID: {attendance_id}")
            
            return attendance_id
            
        except Exception as e:
            print(f"❌ Error recording student attendance: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def show_today_student_attendance(self, student_id: Optional[str] = None):
        """Show today's student attendance - now properly defined"""
        if not self.is_connected():
            print("❌ Database not connected")
            return
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            if student_id:
                cursor.execute("""
                    SELECT sa.*, s.full_name, u.department
                    FROM student_attendance sa
                    JOIN students s ON sa.student_id = s.student_id
                    JOIN users u ON sa.user_id = u.user_id
                    WHERE sa.student_id = %s 
                    AND DATE(sa.attendance_date) = CURRENT_DATE
                    ORDER BY sa.attendance_time DESC
                """, (student_id,))
            else:
                cursor.execute("""
                    SELECT sa.*, s.full_name, u.department
                    FROM student_attendance sa
                    JOIN students s ON sa.student_id = s.student_id
                    JOIN users u ON sa.user_id = u.user_id
                    WHERE DATE(sa.attendance_date) = CURRENT_DATE
                    ORDER BY sa.attendance_time DESC
                    LIMIT 10
                """)
            
            records = cursor.fetchall()
            
            if records:
                print("\n📊 TODAY'S STUDENT ATTENDANCE:")
                print("-" * 80)
                for record in records:
                    time_str = record['attendance_time'].strftime('%H:%M:%S') if record['attendance_time'] else "N/A"
                    print(f"👤 {record['full_name']} (Student ID: {record['student_id']})")
                    print(f"   Type: {record['attendance_type']} at {time_str}")
                    print(f"   Status: {record['attendance_status']} | Confidence: {record.get('confidence_score', 0):.2f}")
                    print("-" * 40)
            else:
                print("📭 No student attendance records for today")
                
        except Exception as e:
            print(f"⚠️ Error fetching student attendance: {e}")
        finally:
            if cursor:
                cursor.close()
    
    def find_user_by_face(self, embedding: np.ndarray, threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """Find user by face embedding"""
        if not self.is_connected():
            print("❌ Database connection is not active")
            return None
    
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            embedding_list = embedding.tolist()
            embedding_str = '[' + ','.join(str(x) for x in embedding_list) + ']'
        
            # Use PostgreSQL vector similarity search
            cursor.execute("""
                WITH face_matches AS (
                    SELECT 
                        ft.id as face_template_id,
                        ft.person_id,
                        ft.person_name,
                        ft.embedding <=> %s as similarity,
                        u.user_id,
                        u.full_name,
                        u.user_type,
                        u.email,
                        u.department
                    FROM face_templates ft
                    JOIN users u ON ft.person_id = u.person_id
                    WHERE ft.embedding <=> %s < %s
                    ORDER BY similarity ASC
                )
                SELECT DISTINCT ON (person_id) *
                FROM face_matches
                ORDER BY person_id, similarity ASC
                LIMIT 1
            """, (embedding_str, embedding_str, 1 - threshold))
        
            result = cursor.fetchone()
        
            if result:
                similarity = 1 - result['similarity']
                result['confidence'] = similarity
                
                # Get additional user info based on user_type
                if result['user_type'] == 'student':
                    cursor.execute("""
                        SELECT s.student_id, s.enrollment_number, s.semester, s.program
                        FROM students s
                        WHERE s.user_id = %s
                    """, (result['user_id'],))
                    student_info = cursor.fetchone()
                    if student_info:
                        result.update(dict(student_info))
                elif result['user_type'] == 'faculty':
                    cursor.execute("""
                        SELECT f.faculty_id, f.designation
                        FROM faculty f
                        WHERE f.user_id = %s
                    """, (result['user_id'],))
                    faculty_info = cursor.fetchone()
                    if faculty_info:
                        result.update(dict(faculty_info))
                
                print(f"✅ Face match found: {result['full_name']} ({result['user_type']})")
                print(f"   Confidence: {similarity:.3f}")
                return dict(result)
        
            return None
        
        except Exception as e:
            print(f"❌ Error in face search: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user details by user_id"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT u.*, 
                       CASE 
                           WHEN u.user_type = 'student' THEN s.student_id
                           WHEN u.user_type = 'faculty' THEN f.faculty_id
                           ELSE NULL
                       END as role_specific_id
                FROM users u
                LEFT JOIN students s ON u.user_id = s.user_id AND u.user_type = 'student'
                LEFT JOIN faculty f ON u.user_id = f.user_id AND u.user_type = 'faculty'
                WHERE u.user_id = %s
            """, (user_id,))
            
            user = cursor.fetchone()
            
            if user:
                user_dict = dict(user)
                
                # Add role-specific details
                if user_dict['user_type'] == 'student':
                    cursor.execute("""
                        SELECT s.enrollment_number, s.semester, s.program, s.batch_year
                        FROM students s
                        WHERE s.user_id = %s
                    """, (user_id,))
                    student_details = cursor.fetchone()
                    if student_details:
                        user_dict.update(dict(student_details))
                
                elif user_dict['user_type'] == 'faculty':
                    cursor.execute("""
                        SELECT f.designation, f.qualification
                        FROM faculty f
                        WHERE f.user_id = %s
                    """, (user_id,))
                    faculty_details = cursor.fetchone()
                    if faculty_details:
                        user_dict.update(dict(faculty_details))
                
                return user_dict
            return None
            
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all registered users"""
        if not self.is_connected():
            return []
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT u.user_id, u.full_name, u.user_type, 
                       u.email, u.department, u.is_active,
                       CASE 
                           WHEN u.user_type = 'student' THEN s.enrollment_number
                           WHEN u.user_type = 'faculty' THEN 'Faculty'
                           ELSE 'Staff'
                       END as role_identifier,
                       COUNT(ft.id) as face_templates_count
                FROM users u
                LEFT JOIN students s ON u.user_id = s.user_id AND u.user_type = 'student'
                LEFT JOIN faculty f ON u.user_id = f.user_id AND u.user_type = 'faculty'
                LEFT JOIN face_templates ft ON u.person_id = ft.person_id
                GROUP BY u.user_id, u.full_name, u.user_type, u.email, 
                         u.department, u.is_active, s.enrollment_number
                ORDER BY u.user_type, u.full_name
            """)
            
            users = cursor.fetchall()
            
            return [dict(user) for user in users]
            
        except Exception as e:
            print(f"❌ Error getting users: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user and associated data (cascade)"""
        if not self.is_connected():
            return False
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            # Get user info before deletion
            user_info = self.get_user_by_id(user_id)
            if not user_info:
                print(f"❌ User {user_id} not found")
                return False
            
            cursor.execute("BEGIN")
            
            # Delete user (cascade will delete related records)
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            
            cursor.execute("COMMIT")
            print(f"✅ User {user_info['full_name']} (ID: {user_id}) deleted successfully")
            return True
            
        except Exception as e:
            if cursor:
                cursor.execute("ROLLBACK")
            print(f"❌ Error deleting user: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get student details by student_id"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT s.*, u.person_id, u.user_type, u.email as user_email
                FROM students s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.student_id = %s
            """, (student_id,))
            
            student = cursor.fetchone()
            
            if student:
                return dict(student)
            return None
            
        except Exception as e:
            print(f"❌ Error getting student: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def get_student_attendance_summary(self, student_id: str, start_date: Optional[str] = None, 
                                      end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get attendance summary for a student"""
        if not self.is_connected():
            return {}
        
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Build query with optional date filters
            query = """
                SELECT 
                    COUNT(*) as total_days,
                    COUNT(CASE WHEN attendance_status = 'present' THEN 1 END) as present_days,
                    COUNT(CASE WHEN attendance_status = 'absent' THEN 1 END) as absent_days,
                    COUNT(CASE WHEN attendance_status = 'late' THEN 1 END) as late_days,
                    AVG(confidence_score) as avg_confidence
                FROM student_attendance
                WHERE student_id = %s
            """
            
            params = [student_id]
            
            if start_date:
                query += " AND attendance_date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND attendance_date <= %s"
                params.append(end_date)
            
            cursor.execute(query, tuple(params))
            summary = cursor.fetchone()
            
            if summary:
                result = dict(summary)
                if result['total_days'] > 0:
                    result['attendance_percentage'] = (result['present_days'] / result['total_days']) * 100
                else:
                    result['attendance_percentage'] = 0
                return result
            return {}
            
        except Exception as e:
            print(f"❌ Error getting attendance summary: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
    
    def create_attendance_session(self, session_data: Dict[str, Any]) -> Optional[int]:
        """Create a new attendance session"""
        if not self.is_connected():
            return None
        
        cursor = None
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO attendance_sessions (
                    session_name, session_type, location_id, room_number,
                    scheduled_start, scheduled_end, course_id, subject_name,
                    faculty_in_charge, session_status, expected_duration_minutes, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING session_id
            """, (
                session_data.get('session_name'),
                session_data.get('session_type', 'lecture'),
                session_data.get('location_id'),
                session_data.get('room_number'),
                session_data.get('scheduled_start'),
                session_data.get('scheduled_end'),
                session_data.get('course_id'),
                session_data.get('subject_name'),
                session_data.get('faculty_in_charge'),
                session_data.get('session_status', 'scheduled'),
                session_data.get('expected_duration_minutes'),
                session_data.get('notes')
            ))
            
            session_id = cursor.fetchone()[0]
            print(f"✅ Attendance session created with ID: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"❌ Error creating attendance session: {e}")
            return None
        finally:
            if cursor:
                cursor.close()