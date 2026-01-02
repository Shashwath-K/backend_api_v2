# routes/auth_routes.py - Authentication routes (login and registration)
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import cv2
import numpy as np
import base64
import io
from PIL import Image
import traceback

# Create blueprint for authentication routes
bp = Blueprint('auth', __name__, url_prefix='/api')

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
    app = current_app
    detector = app.config.get('detector')
    
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
        traceback.print_exc()
        return None, None, None

@bp.route('/login', methods=['POST'])
def face_login():
    """Handle face login request"""
    try:
        app = current_app
        detector = app.config.get('detector')
        encoder = app.config.get('encoder')
        db = app.config.get('db')
        
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
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@bp.route('/register', methods=['POST'])
def register_user():
    """Handle user registration with face"""
    try:
        app = current_app
        detector = app.config.get('detector')
        encoder = app.config.get('encoder')
        db = app.config.get('db')
        
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
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500