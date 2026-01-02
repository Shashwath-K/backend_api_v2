# server.py - Simplified Flask API Server
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
from datetime import datetime
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define mobile app path (go up one level, then into face_Recognition)
MOBILE_APP_PATH = os.path.join(os.path.dirname(__file__), '..', 'face_Recognition')
print(f"📱 Mobile app path: {MOBILE_APP_PATH}")
print(f"📱 Path exists: {os.path.exists(MOBILE_APP_PATH)}")

print("🔧 Initializing face recognition system...")

# Try to import modules
try:
    from detector import FaceDetector
    print("✅ FaceDetector imported")
except ImportError as e:
    print(f"⚠️ Could not import FaceDetector: {e}")
    FaceDetector = None

try:
    from encoder import FaceEncoder
    print("✅ FaceEncoder imported")
except ImportError as e:
    print(f"⚠️ Could not import FaceEncoder: {e}")
    FaceEncoder = None

try:
    from database import DatabaseManager
    print("✅ DatabaseManager imported")
except ImportError as e:
    print(f"⚠️ Could not import DatabaseManager: {e}")
    DatabaseManager = None

# Create a simple Flask app without static folder
try:
    app = Flask(__name__)
    CORS(app)
    print("✅ Flask app initialized")
except Exception as e:
    print(f"❌ Error initializing Flask: {e}")
    # Try alternative initialization
    app = Flask("FaceRecognitionAPI")
    CORS(app)

# Global variables for components
detector = None
encoder = None
db = None

def initialize_components():
    """Initialize face recognition components"""
    global detector, encoder, db
    
    print("🔧 Initializing components...")
    
    # Initialize detector
    if FaceDetector is not None:
        try:
            detector = FaceDetector(method="haar")
            print("✅ Face detector initialized")
        except Exception as e:
            print(f"⚠️ Face detector initialization failed: {e}")
            detector = None
    else:
        print("⚠️ FaceDetector class not available")
        detector = None
    
    # Initialize encoder
    if FaceEncoder is not None:
        try:
            encoder = FaceEncoder(use_deepface=False)
            print("✅ Face encoder initialized")
        except Exception as e:
            print(f"⚠️ Face encoder initialization failed: {e}")
            encoder = None
    else:
        print("⚠️ FaceEncoder class not available")
        encoder = None
    
    # Initialize database
    if DatabaseManager is not None:
        try:
            db = DatabaseManager(
                host="localhost",
                database="attendance_db",
                user="postgres",
                password="root",
                port=5432
            )
            
            if hasattr(db, 'is_connected') and callable(db.is_connected):
                if db.is_connected():
                    print("✅ Database connected")
                else:
                    print("⚠️ Database connection failed")
            else:
                print("✅ Database manager created")
        except Exception as e:
            print(f"⚠️ Database initialization failed: {e}")
            db = None
    else:
        print("⚠️ DatabaseManager class not available")
        db = None
    
    print("🎉 Component initialization completed")

def base64_to_cvimage(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data:image/jpeg;base64, prefix if present
        if isinstance(base64_string, str) and base64_string.startswith('data:'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL to numpy array
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        
        return image_np
        
    except Exception as e:
        print(f"❌ Error converting base64 to image: {e}")
        return None

def process_face_image(image_np):
    """Process image to detect and extract face"""
    if image_np is None:
        return None, None, None
    
    if detector is None:
        print("❌ Face detector not available")
        return None, None, None
    
    try:
        # Detect faces
        faces = detector.detect_faces(image_np)
        
        if len(faces) == 0:
            print("❌ No faces detected in image")
            return None, None, None
        
        # Use the largest face
        face_box = max(faces, key=lambda box: box[2] * box[3])
        x, y, w, h = face_box
        
        # Extract face
        if hasattr(detector, 'extract_face'):
            face_img = detector.extract_face(image_np, face_box)
        else:
            # Simple face extraction
            face_img = image_np[y:y+h, x:x+w]
            if face_img.size > 0:
                face_img = cv2.resize(face_img, (160, 160))
                face_img = face_img.astype('float32')
                face_img = (face_img - 127.5) / 127.5
        
        if face_img is None:
            print("❌ Could not extract face from image")
            return None, None, None
        
        # Convert normalized face back to uint8 for display
        if face_img.dtype == np.float32:
            face_display = ((face_img + 1.0) * 127.5).astype(np.uint8)
        else:
            face_display = face_img.astype(np.uint8) if face_img.dtype != np.uint8 else face_img
        
        return face_img, face_box, face_display
        
    except Exception as e:
        print(f"❌ Error processing face image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ==================== ROUTES ====================

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'detector': detector is not None,
            'encoder': encoder is not None,
            'database': db is not None
        },
        'version': '1.0.0',
        'mobile_app_path': MOBILE_APP_PATH,
        'mobile_app_exists': os.path.exists(MOBILE_APP_PATH)
    })

# Face login endpoint
@app.route('/api/login', methods=['POST'])
def face_login():
    """Handle face login request"""
    try:
        if detector is None or encoder is None:
            return jsonify({
                'success': False,
                'error': 'Face recognition system not fully initialized',
                'debug': {
                    'detector_available': detector is not None,
                    'encoder_available': encoder is not None
                }
            }), 503
        
        data = request.json
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        print("🔄 Processing login request...")
        
        # Convert base64 to image
        image_np = base64_to_cvimage(data['image'])
        if image_np is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            }), 400
        
        # Process face
        face_img, face_box, face_display = process_face_image(image_np)
        if face_img is None:
            return jsonify({
                'success': False,
                'error': 'No face detected or could not process'
            }), 400
        
        print("✅ Face detected and extracted")
        
        # Generate embedding
        embedding = encoder.get_embedding(face_img)
        print(f"📏 Generated embedding with {len(embedding)} dimensions")
        
        # Check if database is available
        if db is None:
            # Simulate a successful login for testing
            return jsonify({
                'success': True,
                'user': {
                    'user_id': 'test_user',
                    'full_name': 'Test User',
                    'user_type': 'student',
                    'confidence': 0.85,
                    'message': 'Database not available - test mode'
                },
                'message': 'Login successful (test mode)'
            })
        
        # Search for user in database
        if hasattr(db, 'find_user_by_face'):
            user_match = db.find_user_by_face(embedding, threshold=0.6)
            
            if user_match:
                print(f"✅ User found: {user_match['full_name']}")
                return jsonify({
                    'success': True,
                    'user': user_match,
                    'message': 'Login successful'
                })
        
        print("❌ No matching user found")
        return jsonify({
            'success': False,
            'message': 'User not recognized'
        })
            
    except Exception as e:
        print(f"❌ Login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

# User registration endpoint
@app.route('/api/register', methods=['POST'])
def register_user():
    """Handle user registration with face"""
    try:
        if detector is None or encoder is None:
            return jsonify({
                'success': False,
                'error': 'Face recognition system not available'
            }), 503
        
        data = request.json
        if not data or 'user_data' not in data:
            return jsonify({
                'success': False,
                'error': 'No user data provided'
            }), 400
        
        user_data = data['user_data']
        face_images = data.get('face_images', [])
        
        print(f"🔄 Processing registration for {user_data.get('full_name', 'Unknown')}")
        
        required_fields = ['full_name', 'user_id', 'user_type']
        for field in required_fields:
            if field not in user_data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
            
        if user_data.get('user_type') == 'student':
            if 'student' not in user_data:
                user_data['student'] = {}

            if 'enrollment_number' not in user_data['student']:
                # Generate a default enrollment number
                user_data['student']['enrollment_number'] = f"ENR{user_data['user_id']}"
                print(f"⚠️ Generated enrollment number: {user_data['student']['enrollment_number']}")
            
            user_data['student']['student_id'] = user_data['user_id']
            user_data['student']['user_id'] = user_data['user_id']
            user_data['student']['full_name'] = user_data['full_name']
        # Process face images
        embeddings = []
        
        for i, img_data in enumerate(face_images):
            print(f"  Processing face image {i+1}/{len(face_images)}...")
            
            image_np = base64_to_cvimage(img_data)
            if image_np is None:
                print(f"    ❌ Failed to decode image {i+1}")
                continue
            
            face_img, face_box, face_display = process_face_image(image_np)
            if face_img is None:
                print(f"    ❌ No face detected in image {i+1}")
                continue
            
            # Generate embedding
            embedding = encoder.get_embedding(face_img)
            
            if len(embedding) != 256:
                print(f"    ⚠️ Image {i+1}: Embedding has {len(embedding)} dimensions")
            
            embeddings.append(embedding)
            print(f"    ✅ Image {i+1} processed successfully")
        
        if not embeddings:
            return jsonify({
                'success': False,
                'error': 'No valid face images provided'
            }), 400
        
        print(f"✅ Processed {len(embeddings)} valid face images")
        
        # If database is available, register user
        if db is not None and hasattr(db, 'register_user'):
            # Prepare face templates data
            person_id = f"USER_{user_data.get('user_id', 'unknown')}"
            face_templates_data = []
            
            for i, embedding in enumerate(embeddings):
                face_template = {
                    'person_id': person_id,
                    'person_name': user_data.get('full_name', 'Unknown'),
                    'embedding': embedding.tolist(),
                    'metadata': {
                        'user_id': user_data.get('user_id', ''),
                        'user_type': user_data.get('user_type', 'student'),
                        'capture_index': i,
                        'capture_time': datetime.now().isoformat()
                    }
                }
                face_templates_data.append(face_template)
            
            # Add person_id to user_data
            user_data['person_id'] = person_id
            
            # Register user in database
            success = db.register_user(user_data, face_templates_data)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Registration successful',
                    'user_id': user_data.get('user_id'),
                    'person_id': person_id,
                    'enrollment_number': user_data.get('student', {}).get('enrollment_number') if user_data.get('user_type') == 'student' else None
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Database registration failed'
                })
        else:
            # Database not available, return success anyway for testing
            return jsonify({
                'success': True,
                'message': 'Registration processed (database not available)',
                'user_id': user_data.get('user_id', 'test_user'),
                'enrollment_number': user_data.get('student', {}).get('enrollment_number') if user_data.get('user_type') == 'student' else None,
                'embeddings_count': len(embeddings),
                'note': 'Running in test mode without database'
            })
            
    except Exception as e:
        print(f"❌ Registration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

# List all users endpoint
@app.route('/api/users', methods=['GET'])
def list_users():
    """Get list of all registered users"""
    try:
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
@app.route('/api/attendance/<student_id>', methods=['GET'])
def get_attendance(student_id):
    """Get attendance records for a student"""
    try:
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
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'records': []
        }), 500
# Simple test endpoint
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify server is running"""
    return jsonify({
        'success': True,
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat(),
        'endpoints': [
            '/api/health',
            '/api/login',
            '/api/register',
            '/api/users',
            '/api/test'
        ]
    })

# Serve test.html page
@app.route('/test.html')
def serve_test_html():
    """Serve the test HTML page"""
    try:
        return send_file('test.html')
    except Exception as e:
        print(f"❌ Error serving test.html: {e}")
        return jsonify({
            'success': False,
            'error': 'test.html not found',
            'message': 'Make sure test.html is in the same directory as server.py'
        }), 404

# Alternative route for test page
@app.route('/test')
def serve_test():
    """Alternative route for test page"""
    try:
        return send_file('test.html')
    except:
        return jsonify({
            'success': False,
            'error': 'Test page not available',
            'instructions': 'Create a test.html file in the server directory'
        }), 404

# ==================== MOBILE APP ROUTES ====================

@app.route('/mobile/<path:filename>')
def serve_mobile_file(filename):
    """Serve mobile app files from face_Recognition folder"""
    try:
        print(f"📱 Serving mobile file: {filename}")
        return send_from_directory(MOBILE_APP_PATH, filename)
    except Exception as e:
        print(f"❌ Error serving mobile file {filename}: {e}")
        return jsonify({
            'success': False,
            'error': f'Mobile file {filename} not found',
            'path': MOBILE_APP_PATH,
            'exists': os.path.exists(MOBILE_APP_PATH)
        }), 404

@app.route('/mobile')
def serve_mobile_index():
    """Serve mobile app index"""
    try:
        print(f"📱 Serving mobile index from: {MOBILE_APP_PATH}")
        if not os.path.exists(MOBILE_APP_PATH):
            return jsonify({
                'success': False,
                'error': 'Mobile app folder not found',
                'path': MOBILE_APP_PATH,
                'current_dir': os.path.dirname(__file__)
            }), 404
        
        index_path = os.path.join(MOBILE_APP_PATH, 'index.html')
        if not os.path.exists(index_path):
            return jsonify({
                'success': False,
                'error': 'index.html not found in mobile app folder',
                'index_path': index_path
            }), 404
        
        return send_from_directory(MOBILE_APP_PATH, 'index.html')
    except Exception as e:
        print(f"❌ Error serving mobile index: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Mobile app not available',
            'details': str(e)
        }), 404

# Also serve files from root for backward compatibility
@app.route('/<path:filename>')
def serve_root_file(filename):
    """Serve files that mobile app expects at root"""
    # Only serve specific file types
    if filename.endswith(('.css', '.js', '.json', '.png', '.jpg', '.jpeg', '.ico')):
        try:
            return send_from_directory(MOBILE_APP_PATH, filename)
        except:
            pass
    return jsonify({'error': 'File not found'}), 404

# Root endpoint
@app.route('/')
def index():
    """API information page"""
    mobile_files = []
    if os.path.exists(MOBILE_APP_PATH):
        try:
            mobile_files = os.listdir(MOBILE_APP_PATH)
        except:
            mobile_files = []
    
    return jsonify({
        'name': 'Face Recognition Attendance API',
        'version': '1.0.0',
        'description': 'API for face recognition-based attendance system',
        'status': 'running',
        'mobile_app': {
            'path': MOBILE_APP_PATH,
            'exists': os.path.exists(MOBILE_APP_PATH),
            'files': mobile_files[:10]  # Show first 10 files
        },
        'endpoints': {
            'GET /': 'API information',
            'GET /api/health': 'System health check',
            'POST /api/login': 'Face login',
            'POST /api/register': 'User registration',
            'GET /api/users': 'List all users',
            'GET /api/attendance/<student_id>': 'Get attendance records',
            'GET /api/test': 'Test endpoint',
            'GET /test.html': 'Test HTML interface',
            'GET /mobile': 'Mobile web app'
        },
        'quick_links': {
            'api_health': 'http://127.0.0.1:5000/api/health',
            'mobile_app': 'http://127.0.0.1:5000/mobile',
            'test_page': 'http://127.0.0.1:5000/test.html'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'success': False}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error', 'success': False}), 500

def main():
    """Main function to start the server"""
    print("\n" + "="*60)
    print("🎭 FACE RECOGNITION ATTENDANCE API SERVER")
    print("="*60)
    
    # Initialize components
    initialize_components()
    
    print("\n🌐 Starting Flask API Server...")
    print("🔧 API Base URL: http://localhost:5000/")
    print("🩺 Health Check: http://localhost:5000/api/health")
    print("🧪 Test Endpoint: http://localhost:5000/api/test")
    print("📋 Test Interface: http://localhost:0:5000/test.html")
    print("📱 Mobile App: http://localhost:5000/mobile")
    print(f"📁 Mobile Path: {MOBILE_APP_PATH}")
    print(f"📁 Path exists: {os.path.exists(MOBILE_APP_PATH)}")
    
    if os.path.exists(MOBILE_APP_PATH):
        print("✅ Mobile app folder found")
        try:
            files = os.listdir(MOBILE_APP_PATH)
            print(f"📄 Found {len(files)} files in mobile app folder")
            for file in files[:5]:  # Show first 5 files
                print(f"   - {file}")
            if len(files) > 5:
                print(f"   ... and {len(files)-5} more")
        except:
            print("⚠️ Could not list mobile app files")
    else:
        print("❌ Mobile app folder NOT found")
        print("   Expected at:", MOBILE_APP_PATH)
        print("   Current directory:", os.path.dirname(__file__))
    
    print("\n📝 Available Endpoints:")
    print("   POST /api/login    - Face login")
    print("   POST /api/register - User registration")
    print("   GET  /api/users    - List all users")
    print("   GET  /api/attendance/<student_id> - Get attendance records")  # ADD THIS LINE
    print("\n🛑 Press Ctrl+C to stop the server")
    print("="*60)
    
    # Start Flask server
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        # Try with different parameters
        print("🔄 Trying alternative configuration...")
        app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':
    main()