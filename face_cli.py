import cv2
import numpy as np
import os
import json
from datetime import datetime, date
import time
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import FaceDetector
from encoder import FaceEncoder
from smart_validator import SmartFaceValidator
from database import DatabaseManager

class FaceRecognitionSystem:
    def __init__(self):
        
        """Initialize face recognition system"""
        print("\n" + "="*50)
        print("🎭 FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*50)
        
        # Check dependencies first
        self.check_dependencies()
        
        # Initialize database connection
        print("\n🔗 Connecting to database...")
        self.db = DatabaseManager(
            host="localhost",
            database="attendance_db",
            user="postgres",
            password="root",
            port=5432
        )
        
        if not self.db.is_connected():
            print("❌ Database connection failed")
            print("Please check PostgreSQL is running and credentials are correct")
            print("Run: sudo systemctl start postgresql")
            print("Then run: psql -U postgres")
            print("And create database: CREATE DATABASE attendance_db;")
            raise ConnectionError("Database connection failed")
        
        # Initialize components
        print("\n🔧 Initializing components...")
        self.detector = FaceDetector(method="haar")  # Default to haar for reliability
        self.encoder = FaceEncoder(use_deepface=False)
        
        # Initialize camera first for validator calibration
        self.camera = None
        self.init_camera()
        
        # Initialize smart validator AFTER camera is ready
        print("\n🤖 Initializing Smart Face Validator...")
        self.validator = SmartFaceValidator(auto_adjust=True)  # Enable auto-adjustment
        
        # Calibrate camera thresholds
        try:
            print("🔧 Calibrating camera for optimal validation...")
            camera_stats = self.validator.calibrate_camera(camera_index=0, test_frames=20)
            print("✅ Camera calibration completed")
            print("\n" + self.validator.get_threshold_summary())
        except Exception as e:
            print(f"⚠️ Camera calibration failed: {e}")
            print("⚠️ Using default validation thresholds")
        
        # Current user
        self.current_user = None
        
        # Check embedding dimensions - MUST be 256 for your schema
        self.verify_embedding_dimensions()
        
        print("\n✅ System ready with calibrated validation thresholds!")
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        print("\n🔍 Checking dependencies...")
        
        try:
            import cv2
            print(f"✅ OpenCV version: {cv2.__version__}")
        except ImportError:
            print("❌ OpenCV is required. Install: pip install opencv-python")
            raise
        
        try:
            import psycopg2
            print("✅ PostgreSQL driver available")
        except ImportError:
            print("❌ psycopg2 required. Install: pip install psycopg2-binary")
            raise
        
        try:
            import numpy as np
            print(f"✅ NumPy version: {np.__version__}")
        except ImportError:
            print("❌ NumPy is required. Install: pip install numpy")
            raise
        
        print("✅ Dependencies check completed")

    def verify_embedding_dimensions(self):
        """Verify and display embedding dimensions - MUST be 256"""
        print("\n📏 Verifying embedding dimensions...")
        
        # Get current embedding dimension from encoder
        test_face = np.zeros((160, 160, 3), dtype=np.uint8)
        test_embedding = self.encoder.get_embedding(test_face)
        current_dim = len(test_embedding)
        
        if current_dim != 256:
            print(f"❌ ERROR: Encoder produces {current_dim}-dimensional embeddings")
            print(f"   Database schema requires exactly 256 dimensions")
            print(f"   Please fix the encoder to output 256 dimensions")
            raise ValueError(f"Encoder must output 256 dimensions, got {current_dim}")
        
        print(f"✅ Encoder produces 256-dimensional embeddings (matches schema)")
        print("✅ Ready for PostgreSQL database operations")
    
    def init_camera(self):
        """Initialize camera with better settings"""
        try:
            # Try different camera indices
            for camera_index in [0, 1, 2]:
                self.camera = cv2.VideoCapture(camera_index)
                if self.camera.isOpened():
                    # Set decent resolution for validation
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camera.set(cv2.CAP_PROP_FPS, 15)
                    print(f"✅ Camera {camera_index} initialized at 640x480")
                    return True
            
            print("❌ No camera found on indices 0, 1, or 2")
            return False
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def get_camera_frame(self, skip_frames: int = 2):
        """Get frame from camera with frame skipping for faster processing"""
        if self.camera is None or not self.camera.isOpened():
            return None
        
        # Skip frames to reduce processing load
        for _ in range(skip_frames):
            self.camera.grab()
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def release_camera(self):
        """Release camera resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
    
    def capture_faces_for_registration(self, person_name: str, required_faces: int = 3):
        print(f"\n📸 Face Capture Instructions:")
        print("1. Look directly at the camera")
        print("2. Press SPACEBAR to capture each photo")
        print("3. Move slightly between captures for better registration")
        print("4. Press ESC to cancel")
        print("\n🎯 Using Smart Validation with calibrated thresholds")
        print("   • Thresholds auto-adjusted for your camera")
        print("   • Real-time quality feedback")
        print("   • Liveness detection active")
    
        face_samples = []
        captured_count = 0
        liveness_failures = 0
        max_liveness_failures = 5  # Increased from 3 for registration
        consecutive_blinks = 0
        blink_required = False
        blink_start_time = 0.0  # Flag to track if blink is required

        print("\n🔧 Current validation thresholds:")
        thresholds = self.validator.current_thresholds
        print(f"   • Blur: > {thresholds.get('blur_threshold', 30.0):.1f}")
        print(f"   • Brightness: {thresholds.get('min_brightness', 0.1):.2f} - {thresholds.get('max_brightness', 0.9):.2f}")
        print(f"   • Contrast: > {thresholds.get('min_contrast', 15.0):.1f}")
        
        cv2.namedWindow("Registration - Smart Validation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Registration - Smart Validation", 800, 600)
        
        while captured_count < required_faces:
            frame = self.get_camera_frame(skip_frames=1)
            if frame is None:
                print("❌ Cannot get camera frame")
                break
            
            display_frame = frame.copy()
            faces = self.detector.detect_faces(frame)
            
            cv2.putText(display_frame, f"Capturing: {person_name}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Samples: {captured_count}/{required_faces}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show liveness status
            liveness_status = "✅" if liveness_failures < max_liveness_failures else "⚠️"
            cv2.putText(display_frame, f"Liveness checks: {liveness_status}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if blink_required:
                cv2.putText(display_frame, "👁️ BLINK NOW for liveness check", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            y_offset = 130
            cv2.putText(display_frame, "Current Thresholds:", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            y_offset += 20
            cv2.putText(display_frame, f"Blur > {thresholds.get('blur_threshold', 30.0):.1f}", 
                       (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
            y_offset += 15
            cv2.putText(display_frame, f"Brightness {thresholds.get('min_brightness', 0.1):.2f}-{thresholds.get('max_brightness', 0.9):.2f}", 
                       (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
            
            if len(faces) == 1:
                face_box = faces[0]
                x, y, w, h = face_box
                face_img, landmarks = self.detector.extract_face_with_landmarks(frame, face_box)

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Perform liveness detection if landmarks are available
                liveness_confidence = 1.0  # Default
                if landmarks is not None and len(landmarks) >= 5:
                    try:
                        # Check if blink is required (every 2nd capture)
                        if captured_count % 2 == 0 and blink_required == False:
                            blink_required = True
                            print("\n👁️ BLINK DETECTION ACTIVE: Please blink now for liveness check")
                            blink_start_time = time.time()
                        
                        # Run liveness check
                        is_live, liveness_confidence, test_results = self.detector.check_liveness(
                            frame, face_box, landmarks, "blink" if blink_required else "basic"
                        )
                        
                        # Display liveness results
                        if test_results:
                            # Show blink detection if active
                            if blink_required and 'blink_detected' in test_results:
                                if test_results['blink_detected']:
                                    cv2.putText(display_frame, "✅ BLINK DETECTED!", 
                                               (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    consecutive_blinks += 1
                                else:
                                    cv2.putText(display_frame, "👁️ Please blink...", 
                                               (x, y - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            # Show face movement detection
                            if 'face_movement' in test_results and test_results['face_movement']:
                                cv2.putText(display_frame, "🔄 Face movement detected", 
                                           (x, y - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Display liveness confidence
                        liveness_color = (0, 255, 0) if liveness_confidence >= 0.7 else (0, 165, 255)
                        liveness_text = f"Liveness: {liveness_confidence:.2f}"
                        cv2.putText(display_frame, liveness_text, (x, y - 140),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, liveness_color, 1)
                        
                        # If blink was required and detected, reset flag
                        if blink_required and 'blink_detected' in test_results and test_results['blink_detected']:
                            if time.time() - blink_start_time < 5:  # 5 second timeout for blink
                                blink_required = False
                                print("✅ Blink detected - liveness check passed")
                            
                    except Exception as e:
                        print(f"⚠️ Liveness check error: {e}")
                        liveness_confidence = 0.8  # Fallback confidence

                # Show real-time validation metrics
                if face_img is not None:
                    if face_img.dtype == np.float32:
                        face_img_display = (face_img * 255).astype(np.uint8)
                    else:
                        face_img_display = face_img.astype(np.uint8) if face_img.dtype != np.uint8 else face_img
                    

                    try:
                        # Blur
                        is_clear, blur_score = self.validator.validate_blur(face_img_display)
                        blur_color = (0, 255, 0) if is_clear else (0, 0, 255)
                        blur_text = f"Blur: {blur_score:.1f} {'✓' if is_clear else '✗'}"
                        cv2.putText(display_frame, blur_text, (x, y - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, blur_color, 1)
                        
                        # Brightness
                        is_bright, bright_score = self.validator.validate_brightness(face_img_display)
                        bright_color = (0, 255, 0) if is_bright else (0, 165, 255)
                        bright_text = f"Bright: {bright_score:.3f} {'✓' if is_bright else '✗'}"
                        cv2.putText(display_frame, bright_text, (x, y - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, bright_color, 1)
                        
                        # Contrast
                        has_contrast, contrast_score = self.validator.validate_contrast(face_img_display)
                        contrast_color = (0, 255, 0) if has_contrast else (0, 165, 255)
                        contrast_text = f"Contrast: {contrast_score:.3f} {'✓' if has_contrast else '✗'}"
                        cv2.putText(display_frame, contrast_text, (x, y - 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, contrast_color, 1)
                        
                    except Exception as e:
                        print(f"Debug metric error: {e}")
                
                # Show instructions
                cv2.putText(display_frame, "SPACE: Capture | ESC: Cancel", (20, display_frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, "For liveness: Move head slightly or blink", 
                           (20, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 200), 1)
                
                # Handle spacebar to capture
                cv2.imshow("Registration - Smart Validation", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 32:  # SPACEBAR
                    if face_img is not None and w > 50 and h > 50:
                        # Check if blink is required but not detected
                        if blink_required and captured_count % 2 == 0:
                            print("❌ Please blink first for liveness check")
                            cv2.putText(display_frame, "❌ BLINK REQUIRED!", 
                                       (x, y - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imshow("Registration - Smart Validation", display_frame)
                            cv2.waitKey(1000)
                            continue
                        
                        # Ensure proper image format
                        if face_img.dtype == np.float32:
                            face_img = (face_img * 255).astype(np.uint8)
                        elif face_img.dtype != np.uint8:
                            face_img = face_img.astype(np.uint8)

                        # Validate using smart validator
                        validation_result = self.validator.validate_face_quality(
                            face_img, 
                            face_box, 
                            frame.shape[:2],
                            debug=False
                        )

                        # Check liveness for registration (more strict than login)
                        liveness_passed = True
                        liveness_warning = ""
                        
                        if landmarks is not None and len(landmarks) >= 5:
                            # For registration, require higher liveness confidence
                            liveness_threshold = 0.8 if captured_count % 2 == 0 else 0.7
                            if liveness_confidence < liveness_threshold:
                                liveness_passed = False
                                liveness_warning = f"Low liveness confidence: {liveness_confidence:.2f} < {liveness_threshold}"
                                liveness_failures += 1
                                if liveness_failures >= max_liveness_failures:
                                    print(f"❌ Too many liveness failures. Registration cancelled.")
                                    cv2.destroyWindow("Registration - Smart Validation")
                                    return None
                        
                        if not validation_result.get('is_valid', True) or not liveness_passed:
                            issues = validation_result.get('issues', [])
                            if not liveness_passed:
                                issues.append(f"liveness_check ({liveness_confidence:.2f})")
                            print(f"❌ Face quality/liveness validation failed")
                            print(f"   Issues: {issues}")
                            
                            # Show failure reasons on screen
                            for i, issue in enumerate(issues):
                                issue_y = y - 180 - (i * 20)
                                cv2.putText(display_frame, f"❌ {issue}", 
                                           (x, issue_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
                            cv2.imshow("Registration - Smart Validation", display_frame)
                            cv2.waitKey(1500)
                        else:
                
                            embedding = self.encoder.get_embedding(face_img)
                            if len(embedding) != 256:
                                print(f"❌ ERROR: Embedding has {len(embedding)} dimensions, expected 256")
                                cv2.putText(display_frame, "❌ EMBEDDING DIM ERROR", 
                                           (x, y - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.imshow("Registration - Smart Validation", display_frame)
                                cv2.waitKey(1000)
                            else:
                                # Success! Save sample
                                face_samples.append({
                                    'image': face_img,
                                    'embedding': embedding,
                                    'box': face_box,
                                    'landmarks': landmarks,
                                    'liveness_confidence': liveness_confidence,
                                    'validation_result': validation_result,
                                    'timestamp': time.time()
                                })
                                captured_count += 1
                                scores = validation_result.get('scores', {})
                                print(f"✅ Captured sample {captured_count}/{required_faces}")
                                print(f"   Embedding dimension: {len(embedding)} (correct!)")
                                print(f"   Liveness confidence: {liveness_confidence:.3f}")
                                if scores:
                                    print(f"   Quality scores:")
                                    for score_name, score_value in scores.items():
                                        if isinstance(score_value, (int, float)):
                                            print(f"     - {score_name}: {score_value:.3f}")
                                
                                # Visual feedback
                                cv2.putText(display_frame, "✅ CAPTURED!", (x, y - 180),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                cv2.putText(display_frame, f"Sample {captured_count}/{required_faces}", 
                                           (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                                
                                # Reset liveness failures on success
                                liveness_failures = 0
                                
                                cv2.imshow("Registration - Smart Validation", display_frame)
                                cv2.waitKey(800)
                    else:
                        print("❌ Face too small, get closer to camera")
                        cv2.putText(display_frame, "FACE TOO SMALL", (x, y - 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        cv2.imshow("Registration - Smart Validation", display_frame)
                        cv2.waitKey(1000)
                elif key == 27:  # ESC
                    print("❌ Registration cancelled")
                    cv2.destroyWindow("Registration - Smart Validation")
                    return None
                
            elif len(faces) > 1:
                cv2.putText(display_frame, "TOO MANY FACES - ONLY ONE PERSON", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Registration - Smart Validation", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("❌ Registration cancelled")
                    cv2.destroyWindow("Registration - Smart Validation")
                    return None
            
            else:
                cv2.putText(display_frame, "NO FACE DETECTED - MOVE CLOSER", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Registration - Smart Validation", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("❌ Registration cancelled")
                    cv2.destroyWindow("Registration - Smart Validation")
                    return None
        
        cv2.destroyWindow("Registration - Smart Validation")
        
        if captured_count == required_faces:
            # Calculate average liveness confidence
            avg_liveness_conf = np.mean([sample['liveness_confidence'] for sample in face_samples])
            print(f"\n✅ Successfully captured {required_faces} face samples")
            print(f"   All embeddings are 256-dimensional")
            print(f"   Average liveness confidence: {avg_liveness_conf:.3f}")
            print(f"   Using calibrated validation thresholds")
            return face_samples
        else:
            print(f"❌ Failed to capture required number of faces")
            return None

    def register_user(self):
        print("\n" + "="*50)
        print("👤 REGISTER NEW USER")
        print("="*50)
    
        # Get user type
        print("\n👥 Select User Type:")
        print("1. Student")
        print("2. Faculty") 
        print("3. Staff")
        print("4. Admin")
        type_choice = input("\nSelect user type (1-4): ").strip()
        user_type_map = {'1': 'student', '2': 'faculty', '3': 'staff', '4': 'admin'}

        if type_choice not in user_type_map:
            print("❌ Invalid selection")
            return False
        user_type = user_type_map[type_choice]

        # Generate appropriate IDs based on type
        if user_type == 'student':
            user_id = input("Enter Student ID (e.g., STU001): ").strip()
            prefix = "STU"
        elif user_type == 'faculty':
            user_id = input("Enter Faculty ID (e.g., FAC001): ").strip()
            prefix = "FAC"
        elif user_type == 'staff':
            user_id = input("Enter Staff ID (e.g., STA001): ").strip()
            prefix = "STA"
        else:  # admin
            user_id = input("Enter Admin ID (e.g., ADM001): ").strip()
            prefix = "ADM"
        
        if not user_id:
            print("❌ User ID is required")
            return False
        
        # Check if user already exists in database
        existing_user = self.db.get_user_by_id(user_id)
        if existing_user:
            print(f"⚠️ User ID '{user_id}' already exists in database")
            choice = input("Overwrite? (y/n): ").lower()
            if choice != 'y':
                return False
        
        print("\n📝 Please enter user details:")
        print("-" * 40)

        full_name = input("Full Name: ").strip()
        if not full_name:
            print("❌ Full name is required")
            return False
    
        email = input("Email Address: ").strip()
    
        phone = input("Phone Number: ").strip()
        dob_str = input("Date of Birth (YYYY-MM-DD) [Optional]: ").strip()
        
        department = input("Department: ").strip()
        role_details = {}
        
        if user_type == 'student':
            role_details = self.collect_student_details(user_id)
        elif user_type == 'faculty':
            role_details = self.collect_faculty_details(user_id)
        elif user_type == 'staff':
            role_details = self.collect_staff_details(user_id)
        elif user_type == 'admin':
            role_details = self.collect_admin_details(user_id)

        if role_details is None:
            print("❌ Registration cancelled - missing required details")
            return False

        self.display_registration_summary(user_id, full_name, user_type, email, department, role_details)

        confirm = input("\n✅ Confirm registration? (y/n): ").lower()
        if confirm != 'y':
            print("❌ Registration cancelled")
            return False
        
        print(f"\n📸 Starting face capture for {full_name}...")
        print("⚠️ IMPORTANT: Each embedding must be exactly 256 dimensions")
        face_samples = self.capture_faces_for_registration(full_name, required_faces=3)
    
        if not face_samples:
            print("❌ Registration failed - could not capture face samples")
            return False
        
        # Prepare data for database
        embeddings = [sample['embedding'] for sample in face_samples]
        person_id = f"{prefix}_{user_id}"
        
        # Prepare user data
        user_data = {
            'user_id': user_id,
            'person_id': person_id,
            'full_name': full_name,
            'user_type': user_type,
            'email': email,
            'phone': phone,
            'date_of_birth': dob_str if dob_str else None,
            'department': department
        }
        
        # Prepare face templates data
        face_templates_data = []
        for i, embedding in enumerate(embeddings):
            # CRITICAL: Ensure embedding is exactly 256 dimensions
            if len(embedding) != 256:
                print(f"❌ ERROR: Embedding {i+1} has {len(embedding)} dimensions, expected 256")
                print("   Registration failed due to dimension mismatch")
                return False
            
            face_template = {
                'person_id': person_id,
                'person_name': full_name,
                'embedding': embedding.tolist(),
                'metadata': {
                    'user_id': user_id,
                    'user_type': user_type,
                    'capture_index': i,
                    'capture_time': datetime.now().isoformat(),
                    'embedding_dim': len(embedding)
                }
            }
            face_templates_data.append(face_template)
        
        # Add role-specific data
        if user_type == 'student' and role_details:
            user_data['student'] = {
                'student_id': user_id,
                'user_id': user_id,
                'full_name': full_name,
                'enrollment_number': role_details.get('enrollment_number', ''),
                'semester': role_details.get('semester'),
                'program': role_details.get('program'),
                'batch_year': role_details.get('batch_year'),
                'email': email,
                'phone': phone,
                'date_of_birth': dob_str if dob_str else None
            }
        elif user_type == 'faculty' and role_details:
            user_data['faculty'] = {
                'faculty_id': user_id,
                'user_id': user_id,
                'designation': role_details.get('designation'),
                'qualification': role_details.get('qualification')
            }
        
        # Save to database
        success = self.db.register_user(user_data, face_templates_data)
        
        if success:
            self.show_registration_complete(user_id, full_name, user_type)
            return True
        else:
            print("❌ Registration failed - database error")
            return False
    
    def show_registration_complete(self, user_id: str, full_name: str, user_type: str):
        print(f"\n🎉 REGISTRATION COMPLETE!")
        print(f"👤 User: {full_name}")
        print(f"🆔 User ID: {user_id}")
        print(f"👥 Type: {user_type.upper()}")
        print(f"📏 Embedding Dimension: 256 (matches schema)")
        print(f"💾 Data saved to PostgreSQL database")
    
        # Create visual confirmation
        colors = {
            'student': (0, 150, 0),  
            'faculty': (0, 100, 200), 
            'staff': (200, 100, 0),
            'admin': (150, 0, 150)
        }
    
        bg_color = colors.get(user_type, (0, 100, 0))
    
        frame = np.zeros((350, 600, 3), dtype=np.uint8)
        frame[:] = bg_color
    
        # Title
        type_titles = {
            'student': "🎓 STUDENT REGISTERED",
            'faculty': "👨‍🏫 FACULTY REGISTERED",
            'staff': "👨‍💼 STAFF REGISTERED",
            'admin': "👨‍💻 ADMIN REGISTERED"
        }
    
        title = type_titles.get(user_type, "USER REGISTERED")
    
        cv2.putText(frame, "✅ " + title, (100, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
        cv2.putText(frame, f"Name: {full_name}", (150, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
        cv2.putText(frame, f"ID: {user_id}", (150, 180),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
        cv2.putText(frame, f"Type: {user_type.upper()}", (150, 210),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
        cv2.putText(frame, "Embedding: 256 dimensions ✓", (120, 250),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 200), 1)
    
        cv2.putText(frame, "Data saved to PostgreSQL", (120, 280),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 200), 1)
    
        cv2.putText(frame, "Press any key to continue...", (160, 320),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
        cv2.imshow("Registration Complete", frame)
        cv2.waitKey(3000)
        cv2.destroyWindow("Registration Complete")
    
    def collect_student_details(self, user_id: str):
        print("\n🎓 Student Details:")
        print("-" * 30)
        enrollment_number = input("Enrollment Number: ").strip()
        if not enrollment_number:
            print("❌ Enrollment number is required for students")
            return None
        semester_input = input("Current Semester (1-8): ").strip()
        try:
            semester = int(semester_input) if semester_input else None
        except ValueError:
            print("❌ Semester must be a number")
            return None
        program = input("Program/Degree: ").strip()
        batch_year = input("Batch Year (e.g., 2024): ").strip()

        return {
            'enrollment_number': enrollment_number,
            'semester': semester,
            'program': program,
            'batch_year': batch_year
        }
    
    def collect_faculty_details(self, user_id: str):
        print("\n👨‍🏫 Faculty Details:")
        print("-" * 30)
        designation = input("Designation: ").strip()
        qualification = input("Qualification: ").strip()

        return {
            'designation': designation,
            'qualification': qualification
        }

    def collect_staff_details(self, user_id: str):
        print("\n👨‍💼 Staff Details:")
        print("-" * 30)
        designation = input("Designation: ").strip()
        return {'designation': designation}
    
    def collect_admin_details(self, user_id: str):
        print("\n👨‍💻 Admin Details:")
        print("-" * 30)
        designation = input("Designation: ").strip()
        return {'designation': designation}
    
    def display_registration_summary(self, user_id, full_name, user_type, email, department, role_details):
        print("\n" + "="*50)
        print("📋 REGISTRATION SUMMARY")
        print("="*50)
        print(f"User ID: {user_id}")
        print(f"Full Name: {full_name}")
        print(f"User Type: {user_type.upper()}")
        print(f"Email: {email if email else '(Not provided)'}")
        print(f"Department: {department if department else '(Not provided)'}")
        if role_details:
            print(f"\n{user_type.upper()} Details:")
            for key, value in role_details.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    def quick_face_login(self, threshold: float = 0.6, timeout: int = 30):
        """
        Face login using database search
        """
        print("\n" + "="*50)
        print("🔐 FACE ATTENDANCE LOGIN")
        print("="*50)
        print("Look at the camera and press SPACE when ready...")
        
        start_time = time.time()
        
        cv2.namedWindow("Face Login - Press SPACE to Login", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Login - Press SPACE to Login", 640, 480)
        
        while time.time() - start_time < timeout:
            frame = self.get_camera_frame(skip_frames=1)
            if frame is None:
                print("❌ Camera error")
                break
            
            display_frame = frame.copy()
            
            # Detect faces
            faces = self.detector.detect_faces(frame)
            
            if len(faces) == 1:
                face_box = faces[0]
                x, y, w, h = face_box
                
                # Draw face box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, "FACE DETECTED - PRESS SPACE", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check for spacebar press
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACEBAR
                    print("\n⏳ Searching in database...")
                    
                    # Extract face
                    face_img = self.detector.extract_face(frame, face_box)
                    if face_img is not None:
                        print("🔍 Performing liveness detection...")
                        _, landmarks = self.detector.extract_face_with_landmarks(frame, face_box)
                        is_live, liveness_confidence, test_results = self.detector.check_liveness(
                        frame, face_box, landmarks, "comprehensive")

                        if not is_live or liveness_confidence < 0.7:
                            print(f"❌ Liveness check failed! Confidence: {liveness_confidence:.3f}")
                            print("⚠️ SPOOF DETECTED - Please use a real face, not a photo/video")
                            cv2.putText(display_frame, "SPOOF DETECTED!", (x, y - 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(display_frame, f"Use real face", (x, y - 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.imshow("Face Login - Press SPACE to Login", display_frame)
                            cv2.waitKey(2000)
                            continue
                        print(f"✅ Liveness check passed! Confidence: {liveness_confidence:.3f}")
                        # Generate embedding (must be 256-dim)
                        query_embedding = self.encoder.get_embedding(face_img)
                        
                        # Verify embedding dimension
                        if len(query_embedding) != 256:
                            print(f"❌ ERROR: Query embedding has {len(query_embedding)} dimensions, expected 256")
                            continue
                        
                        # Search in database using the DatabaseManager
                        match_result = self.db.find_user_by_face(query_embedding, threshold)
                        
                        if match_result and match_result.get('confidence', 0) >= threshold:
                            # Login successful
                            self.current_user = match_result
                            
                            cv2.destroyWindow("Face Login - Press SPACE to Login")
                            
                            # Show login success
                            self.show_login_success_with_details(match_result)
                            
                            # Record attendance for students
                            if match_result['user_type'] == 'student':
                                print("\n📝 Recording student attendance...")
                                
                                # Prepare attendance data
                                attendance_data = {
                                    'student_id': match_result.get('student_id', match_result['user_id']),
                                    'user_id': match_result['user_id'],
                                    'person_id': match_result['person_id'],
                                    'face_template_id': match_result.get('face_template_id'),
                                    'confidence_score': match_result.get('confidence', 0.9),
                                    'attendance_type': 'class',
                                    'attendance_status': 'present',
                                    'authentication_method': 'face'
                                }
                                
                                attendance_id = self.db.record_student_attendance(attendance_data)
                                if attendance_id:
                                    self.current_user['attendance_id'] = attendance_id
                            
                            return self.current_user
                        else:
                            # Login failed
                            confidence = match_result.get('confidence', 0) if match_result else 0
                            print(f"❌ Login failed - Confidence: {confidence:.3f}")
                            
                            # Show failure on screen
                            cv2.putText(display_frame, "LOGIN FAILED", (x, y - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(display_frame, f"Score: {confidence:.2f}", 
                                       (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.imshow("Face Login - Press SPACE to Login", display_frame)
                            cv2.waitKey(1000)
            elif len(faces) > 1:
                cv2.putText(display_frame, "TOO MANY FACES - ONLY ONE PERSON", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "NO FACE DETECTED - MOVE CLOSER", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show timer
            elapsed = int(time.time() - start_time)
            remaining = timeout - elapsed
            timer_text = f"Time: {remaining}s | Threshold: {threshold}"
            cv2.putText(display_frame, timer_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show instructions
            cv2.putText(display_frame, "SPACE: Try Login | ESC: Cancel", (20, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Face Login - Press SPACE to Login", display_frame)
            
            # Check for escape key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("❌ Login cancelled")
                break
        
        cv2.destroyWindow("Face Login - Press SPACE to Login")
        
        if time.time() - start_time >= timeout:
            print("❌ Login timeout")
        
        return None
    
    def show_login_success_with_details(self, user_info: dict):
        """Display login success with user details"""
        user_type = user_info.get('user_type', 'user')
        full_name = user_info.get('full_name', 'User')
        confidence = user_info.get('confidence', 0.0)

        print(f"\n" + "="*50)
        print(f"🎉 LOGIN SUCCESSFUL!")
        print("="*50)

        greetings = {
            'student': f"🎓 Welcome back, {full_name}!",
            'faculty': f"👨‍🏫 Welcome, Professor {full_name}!",
            'staff': f"👨‍💼 Welcome, {full_name}!",
            'admin': f"👨‍💻 Welcome, Administrator {full_name}!"
        }

        greeting = greetings.get(user_type, f"👤 Welcome, {full_name}!")
        print(greeting)
        print(f"📊 Confidence Score: {confidence:.3f}")
        print(f"👥 Role: {user_type.upper()}")
        print(f"⏰ Login Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if 'department' in user_info and user_info['department']:
            print(f"🏫 Department: {user_info['department']}")
        
        # Show role-specific info
        if user_type == 'student':
            if 'enrollment_number' in user_info and user_info['enrollment_number']:
                print(f"📚 Enrollment No: {user_info['enrollment_number']}")
            if 'program' in user_info and user_info['program']:
                print(f"🎓 Program: {user_info['program']}")
            if 'semester' in user_info and user_info['semester']:
                print(f"📅 Semester: {user_info['semester']}")
            if 'batch_year' in user_info and user_info['batch_year']:
                print(f"📅 Batch Year: {user_info['batch_year']}")
            if 'attendance_id' in user_info:
                print(f"✅ Attendance Recorded: ID {user_info['attendance_id']}")
        elif user_type == 'faculty' and 'designation' in user_info:
            print(f"👨‍🏫 Designation: {user_info['designation']}")

        colors = {
            'student': (0, 100, 0),
            'faculty': (0, 50, 100),  
            'staff': (100, 50, 0),
            'admin': (100, 0, 50)
        }

        bg_color = colors.get(user_type, (0, 100, 0))
    
        success_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        success_frame[:] = bg_color

        titles = {
            'student': "🎓 STUDENT ACCESS GRANTED",
            'faculty': "👨‍🏫 FACULTY ACCESS GRANTED", 
            'staff': "👨‍💼 STAFF ACCESS GRANTED",
            'admin': "👨‍💻 ADMIN ACCESS GRANTED"
        }
    
        title = titles.get(user_type, "ACCESS GRANTED")
    
        cv2.putText(success_frame, "✓ " + title, (80, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.putText(success_frame, f"Welcome, {full_name}!", (150, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
        cv2.putText(success_frame, f"Role: {user_type.upper()}", (200, 180),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
        cv2.putText(success_frame, f"Confidence: {confidence:.3f}", (200, 210),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Show additional info for students
        if user_type == 'student' and 'attendance_id' in user_info:
            cv2.putText(success_frame, f"Attendance Recorded ✓", (200, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 200), 1)
        
        cv2.putText(success_frame, "Press any key to continue...", (160, 320),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
        cv2.imshow("Login Successful", success_frame)
        cv2.waitKey(3000)
        cv2.destroyWindow("Login Successful")
    
    def logout(self):
        """Logout current user"""
        if self.current_user:
            print(f"\n👋 Logging out {self.current_user['full_name']}...")
            
            # Record check-out attendance for students
            if self.current_user.get('user_type') == 'student':
                print(f"✅ {self.current_user['full_name']} logged out (student)")
            else:
                print(f"✅ {self.current_user['full_name']} logged out")
            
            self.current_user = None
        else:
            print("❌ No user is currently logged in")
    
    def list_users(self):
        print("\n" + "="*50)
        print("📋 REGISTERED USERS FROM DATABASE")
        print("="*50)
        
        users = self.db.get_all_users()
        
        if not users:
            print("No users registered in database")
            return
        
        by_type = {'student': [], 'faculty': [], 'staff': [], 'admin': []}
        
        for user in users:
            user_type = user.get('user_type', 'unknown')
            if user_type in by_type:
                by_type[user_type].append(user)
        
        type_labels = {
            'student': '🎓 STUDENTS',
            'faculty': '👨‍🏫 FACULTY',
            'staff': '👨‍💼 STAFF',
            'admin': '👨‍💻 ADMIN'
        }
        
        total = 0
        for user_type, label in type_labels.items():
            users_of_type = by_type[user_type]
            if users_of_type:
                print(f"\n{label} ({len(users_of_type)}):")
                print("-" * 40)
                for user in users_of_type:
                    total += 1
                    print(f"  • {user['full_name']} (ID: {user['user_id']})")
                    print(f"    Email: {user.get('email', 'N/A')}")
                    print(f"    Department: {user.get('department', 'N/A')}")
                    print(f"    Status: {'Active' if user.get('is_active', True) else 'Inactive'}")
                    if user.get('face_templates_count', 0) > 0:
                        print(f"    Face Templates: {user['face_templates_count']} (256-dim each)")
        
        print(f"\n📊 Total Users in Database: {total}")
        print(f"📏 Embedding Dimension: 256 (fixed)")

    def delete_user(self):
        """Delete a user from database"""
        print("\n" + "="*50)
        print("🗑️ DELETE USER FROM DATABASE")
        print("="*50)
        
        self.list_users()  # Show current users
        
        user_id = input("\nEnter User ID to delete: ").strip()
        
        if not user_id:
            print("❌ No user ID provided")
            return
        
        confirm = input(f"Delete user {user_id}? This action cannot be undone. (y/n): ").lower()
        
        if confirm == 'y':
            success = self.db.delete_user(user_id)
            if success:
                print(f"✅ User {user_id} deleted from database")
                # Clear current user if they were deleted
                if self.current_user and self.current_user.get('user_id') == user_id:
                    self.current_user = None
        else:
            print("❌ Deletion cancelled")
    
    def show_current_user(self):
        """Show currently logged in user"""
        print("\n" + "="*50)
        print("👤 CURRENT USER")
        print("="*50)
        
        if self.current_user:
            print(f"Name: {self.current_user.get('full_name', 'N/A')}")
            print(f"User ID: {self.current_user.get('user_id', 'N/A')}")
            print(f"Role: {self.current_user.get('user_type', 'N/A').upper()}")
            print(f"Login Confidence: {self.current_user.get('confidence', 0.0):.3f}")
            
            if self.current_user.get('user_type') == 'student':
                if 'enrollment_number' in self.current_user:
                    print(f"Enrollment No: {self.current_user['enrollment_number']}")
                if 'program' in self.current_user:
                    print(f"Program: {self.current_user['program']}")
                if 'semester' in self.current_user:
                    print(f"Semester: {self.current_user['semester']}")
                if 'batch_year' in self.current_user:
                    print(f"Batch Year: {self.current_user['batch_year']}")
            elif self.current_user.get('user_type') == 'faculty' and 'designation' in self.current_user:
                print(f"Designation: {self.current_user['designation']}")
            
            if 'attendance_id' in self.current_user:
                print(f"Check-in ID: {self.current_user['attendance_id']}")
        else:
            print("No user is currently logged in")
    
    def test_camera(self):
        """Enhanced camera test with threshold visualization"""
        print("\n" + "="*50)
        print("📷 CAMERA TEST WITH VALIDATION THRESHOLDS")
        print("="*50)
        
        if self.camera is None or not self.camera.isOpened():
            print("❌ Camera not initialized")
            return False
        
        print("Testing camera with current validation thresholds...")
        print("\n🔧 Current validation thresholds:")
        thresholds = self.validator.current_thresholds
        print(f"   • Blur: > {thresholds.get('blur_threshold', 30.0):.1f}")
        print(f"   • Brightness: {thresholds.get('min_brightness', 0.1):.2f} - {thresholds.get('max_brightness', 0.9):.2f}")
        print(f"   • Contrast: > {thresholds.get('min_contrast', 15.0):.1f}")
        
        cv2.namedWindow("Camera Test with Validation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Test with Validation", 800, 600)
        
        for i in range(100):  # Test for 100 frames
            frame = self.get_camera_frame()
            if frame is None:
                print("❌ Cannot get camera frame")
                cv2.destroyWindow("Camera Test with Validation")
                return False
            
            # Convert to grayscale for metrics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            frame_blur = float(laplacian.var())
            gray_mean_float = float(np.mean(gray))
            frame_brightness = gray_mean_float / 255.0
            frame_contrast = gray.std()
            
            # Display metrics
            y_offset = 30
            cv2.putText(frame, f"Camera Test - Frame {i+1}/100", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            # Blur status
            blur_color = (0, 255, 0) if frame_blur > thresholds.get('blur_threshold', 30.0) else (0, 0, 255)
            cv2.putText(frame, f"Blur: {frame_blur:.1f} / {thresholds.get('blur_threshold', 30.0):.1f}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blur_color, 2)
            y_offset += 25
            
            # Brightness status
            is_bright = (thresholds.get('min_brightness', 0.1) <= frame_brightness <= thresholds.get('max_brightness', 0.9))
            bright_color = (0, 255, 0) if is_bright else (0, 165, 255)
            cv2.putText(frame, f"Brightness: {frame_brightness:.3f}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bright_color, 2)
            y_offset += 25
            cv2.putText(frame, f"Range: [{thresholds.get('min_brightness', 0.1):.2f}-{thresholds.get('max_brightness', 0.9):.2f}]", 
                       (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bright_color, 1)
            y_offset += 20
            
            # Contrast status
            contrast_color = (0, 255, 0) if frame_contrast > thresholds.get('min_contrast', 15.0) else (0, 165, 255)
            cv2.putText(frame, f"Contrast: {frame_contrast:.1f} / {thresholds.get('min_contrast', 15.0):.1f}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, contrast_color, 2)
            
            # Show instructions
            cv2.putText(frame, "Press ESC to skip test", (20, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Camera Test with Validation", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("Camera test skipped")
                break
        
        cv2.destroyWindow("Camera Test with Validation")
        print("\n✅ Camera test completed with threshold validation")
        print("📊 Final metrics:")
        print(f"   • Blur: {frame_blur:.1f} (threshold: {thresholds.get('blur_threshold', 30.0):.1f})")
        print(f"   • Brightness: {frame_brightness:.3f} (range: [{thresholds.get('min_brightness', 0.1):.2f}-{thresholds.get('max_brightness', 0.9):.2f}])")
        print(f"   • Contrast: {frame_contrast:.1f} (threshold: {thresholds.get('min_contrast', 15.0):.1f})")
        return True
    
    def recalibrate_camera(self):
        """Recalibrate camera thresholds"""
        print("\n" + "="*50)
        print("🔧 RECALIBRATE CAMERA THRESHOLDS")
        print("="*50)
        
        try:
            print("Recalibrating camera...")
            self.validator.calibrate_camera(camera_index=0, test_frames=30)
            print("\n✅ Camera recalibrated successfully!")
            print(self.validator.get_threshold_summary())
        except Exception as e:
            print(f"❌ Recalibration failed: {e}")
    
    def show_attendance(self):
        """Show attendance records"""
        print("\n" + "="*50)
        print("📊 STUDENT ATTENDANCE RECORDS")
        print("="*50)
        
        print("\nOptions:")
        print("1. Show today's student attendance")
        print("2. Show my attendance (if logged in as student)")
        print("3. Show all student attendance")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            self.db.show_today_student_attendance()
        elif choice == "2":
            if self.current_user and self.current_user.get('user_type') == 'student':
                student_id = self.current_user.get('student_id', self.current_user['user_id'])
                self.db.show_today_student_attendance(student_id)
            else:
                print("❌ Please login as a student first")
        elif choice == "3":
            # Show all student attendance
            self.db.show_today_student_attendance()
        else:
            print("❌ Invalid choice")
    
    def run(self):
        """Run the interactive system"""
        try:
            # Show calibration info at start
            print("\n🔧 Smart Face Validator Initialized:")
            print(self.validator.get_threshold_summary())
            
            # Test camera first
            if not self.test_camera():
                print("❌ Camera test failed. Please check your camera connection.")
                return
            
            while True:
                print("\n" + "="*50)
                print("📝 FACE RECOGNITION ATTENDANCE SYSTEM")
                print("="*50)
                print("🔧 Using calibrated validation thresholds")
                
                # Show current user if logged in
                if self.current_user:
                    user_type = self.current_user.get('user_type', 'user')
                    user_name = self.current_user.get('full_name', 'User')
                    
                    user_icons = {
                        'student': '🎓',
                        'faculty': '👨‍🏫',
                        'staff': '👨‍💼',
                        'admin': '👨‍💻'
                    }
                    
                    icon = user_icons.get(user_type, '👤')
                    print(f"{icon} Currently logged in: {user_name} ({user_type.upper()})")
                    
                    # Show attendance info for students
                    if user_type == 'student' and 'attendance_id' in self.current_user:
                        print(f"   📝 Attendance ID: {self.current_user['attendance_id']}")
                    
                    print("-" * 50)
                
                print("1. Register New User (Camera)")
                print("2. Face Login & Check-in")
                print("3. Logout")
                print("4. List Registered Users")
                print("5. Delete User")
                print("6. Show Current User")
                print("7. Show Student Attendance")
                print("8. Test Camera & Validation")
                print("9. Recalibrate Camera Thresholds")
                print("0. Exit")
                print("="*50)
                
                choice = input("\nSelect option (0-9): ").strip()
                
                if choice == "1":
                    self.register_user()
                elif choice == "2":
                    if self.current_user:
                        print("❌ Please logout first")
                    else:
                        self.quick_face_login(threshold=0.6, timeout=20)
                elif choice == "3":
                    self.logout()
                elif choice == "4":
                    self.list_users()
                elif choice == "5":
                    self.delete_user()
                elif choice == "6":
                    self.show_current_user()
                elif choice == "7":
                    self.show_attendance()
                elif choice == "8":
                    self.test_camera()
                elif choice == "9":
                    self.recalibrate_camera()
                elif choice == "0":
                    print("\n👋 Thank you for using Face Recognition Attendance System!")
                    print("Goodbye!\n")
                    break
                else:
                    print("❌ Invalid choice. Please select 0-9.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.release_camera()
            self.db.close()  # Close database connection

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🎭 FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*60)
    print("\nIMPORTANT SYSTEM REQUIREMENTS:")
    print("📏 EMBEDDING DIMENSION: MUST be exactly 256")
    print("💾 DATABASE SCHEMA: VECTOR(256) field")
    print("🎯 SMART VALIDATION: Auto-adjusts thresholds for your camera")
    print("\nINSTRUCTIONS:")
    print("1. Make sure camera is connected")
    print("2. System will auto-calibrate thresholds for your environment")
    print("3. Sit in a well-lit area")
    print("4. Look directly at the camera")
    print("5. For registration: Press SPACEBAR to capture each photo")
    print("6. For login: Press SPACEBAR when your face is detected")
    print("7. Student attendance is automatically recorded on login")
    print("8. Data is stored in PostgreSQL with 256-dim embeddings")
    
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        print(f"\n❌ System initialization failed: {e}")

if __name__ == "__main__":
    main()