import cv2
import numpy as np
import os
import sys
from typing import List, Tuple, Optional, Union
import time
from enum import Enum

class LivenessTest(Enum):
    BLINK = "blink"
    HEAD_MOVEMENT = "head_movement"
    MOUTH_MOVEMENT = "mouth_movement"
    COLOR_SATURATION = "color_saturation"
    TEXTURE_ANALYSIS = "texture_analysis"
    MOTION_ANALYSIS = "motion_analysis"

class LivenessDetector:
    """Detect liveness to prevent spoofing with photos/videos"""
    
    def __init__(self):
        self.eye_aspect_ratio_threshold = 0.25
        self.mouth_aspect_ratio_threshold = 0.30
        self.blink_threshold = 0.18
        self.motion_threshold = 2.0
        self.color_saturation_threshold = 0.3
        self.texture_variance_threshold = 100
        
        # State tracking
        self.eye_state_history = []
        self.mouth_state_history = []
        self.head_pose_history = []
        self.frame_history = []
        self.max_history = 10
        
        # Face landmark indices (approximate - 68-point model)
        self.FACE_LANDMARKS_68 = {
            'left_eye': list(range(36, 42)),
            'right_eye': list(range(42, 48)),
            'nose_bridge': list(range(27, 31)),
            'nose_tip': [30, 31, 32, 33, 34, 35],
            'mouth_outer': list(range(48, 60)),
            'mouth_inner': list(range(60, 68))
        }
    
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        if len(eye_points) < 6:
            return 0.0
        
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def calculate_mouth_aspect_ratio(self, mouth_points):
        """Calculate Mouth Aspect Ratio (MAR) for mouth movement detection"""
        if len(mouth_points) < 12:
            return 0.0
        
        # Vertical distances
        v1 = np.linalg.norm(mouth_points[2] - mouth_points[10])
        v2 = np.linalg.norm(mouth_points[4] - mouth_points[8])
        
        # Horizontal distance
        h = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        if h == 0:
            return 0.0
        
        mar = (v1 + v2) / (2.0 * h)
        return mar
    
    def estimate_head_pose(self, face_landmarks, image_size):
        """Estimate head pose (pitch, yaw, roll)"""
        if face_landmarks is None or len(face_landmarks) < 5:
            return (0, 0, 0)
        
        # Simple head pose estimation using face landmarks
        # This is a simplified version - in production, use solvePnP
        
        # Get key points
        left_eye = face_landmarks[0] if len(face_landmarks) > 0 else np.array([0, 0])
        right_eye = face_landmarks[1] if len(face_landmarks) > 1 else np.array([0, 0])
        nose = face_landmarks[2] if len(face_landmarks) > 2 else np.array([0, 0])
        left_mouth = face_landmarks[3] if len(face_landmarks) > 3 else np.array([0, 0])
        right_mouth = face_landmarks[4] if len(face_landmarks) > 4 else np.array([0, 0])
        
        # Calculate simple angles (approximate)
        eye_center = (left_eye + right_eye) / 2
        
        # Yaw (left-right rotation)
        eye_distance = np.linalg.norm(right_eye - left_eye)
        nose_offset = np.linalg.norm(nose - eye_center)
        
        if eye_distance > 0:
            yaw_angle = np.arctan2(nose_offset, eye_distance) * 180 / np.pi
        else:
            yaw_angle = 0
        
        # Pitch (up-down rotation)
        mouth_center = (left_mouth + right_mouth) / 2
        vertical_distance = np.linalg.norm(mouth_center - eye_center)
        
        if vertical_distance > 0:
            pitch_angle = np.arctan2(nose[1] - eye_center[1], vertical_distance) * 180 / np.pi
        else:
            pitch_angle = 0
        
        # Roll (tilt)
        if right_eye[0] != left_eye[0]:
            roll_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
        else:
            roll_angle = 0
        
        return (pitch_angle, yaw_angle, roll_angle)
    
    def detect_blink(self, face_landmarks):
        """Detect eye blink using Eye Aspect Ratio"""
        if face_landmarks is None or len(face_landmarks) < 48:
            return False
        
        # Get eye points (simplified - using 5-point landmarks)
        left_eye = np.array([face_landmarks[0]])
        right_eye = np.array([face_landmarks[1]])
        
        # For 5-point model, we approximate blink detection
        # In production with 68-point model, use proper EAR calculation
        
        # Simplified blink detection for 5-point landmarks
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        # Store in history
        self.eye_state_history.append(eye_distance)
        if len(self.eye_state_history) > self.max_history:
            self.eye_state_history.pop(0)
        
        # Check for sudden decrease in eye distance (blink)
        if len(self.eye_state_history) >= 3:
            current = self.eye_state_history[-1]
            previous = self.eye_state_history[-2]
            
            if previous > 0 and current > 0:
                ratio = current / previous
                if ratio < self.blink_threshold:
                    return True
        
        return False
    
    def detect_mouth_movement(self, face_landmarks):
        """Detect mouth opening/closing"""
        if face_landmarks is None or len(face_landmarks) < 5:
            return False
        
        # For 5-point model, use mouth corners
        left_mouth = face_landmarks[3]
        right_mouth = face_landmarks[4]
        
        mouth_width = np.linalg.norm(right_mouth - left_mouth)
        
        # Store in history
        self.mouth_state_history.append(mouth_width)
        if len(self.mouth_state_history) > self.max_history:
            self.mouth_state_history.pop(0)
        
        # Check for significant mouth width change
        if len(self.mouth_state_history) >= 3:
            current = self.mouth_state_history[-1]
            avg_previous = np.mean(self.mouth_state_history[-3:-1])
            
            if avg_previous > 0:
                ratio = current / avg_previous
                if ratio > 1.5 or ratio < 0.7:  # Significant change
                    return True
        
        return False
    
    def detect_head_movement(self, face_landmarks, image_size):
        """Detect head rotation movements"""
        head_pose = self.estimate_head_pose(face_landmarks, image_size)
        
        # Store in history
        self.head_pose_history.append(head_pose)
        if len(self.head_pose_history) > self.max_history:
            self.head_pose_history.pop(0)
        
        # Check for head movement
        if len(self.head_pose_history) >= 3:
            current_yaw = head_pose[1]
            previous_yaw = self.head_pose_history[-2][1]
            
            if abs(current_yaw - previous_yaw) > 10:  # Significant yaw change
                return True
        
        return False
    
    def analyze_color_saturation(self, face_region):
        """Analyze color saturation to detect printed photos (usually less saturated)"""
        if face_region is None or face_region.size == 0:
            return 1.0
        
        # Convert to HSV
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Calculate average saturation
        avg_saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        return avg_saturation
    
    def analyze_texture(self, face_region):
        """Analyze texture to detect 2D vs 3D faces"""
        if face_region is None or face_region.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture variance (3D faces have more texture variation)
        texture_variance = np.var(gray)
        
        # Normalize
        normalized_variance = texture_variance / 10000.0
        
        return min(normalized_variance, 1.0)
    
    def analyze_motion(self, current_frame, face_box):
        if current_frame is None or face_box is None:
            return 0.0
    
        x, y, w, h = face_box
    
    # Ensure valid region
        x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
        if w <= 0 or h <= 0:
            return 0.0
    
    # Extract face region
        face_region = current_frame[y:y+h, x:x+w]
        if face_region.size == 0:
            return 0.0
    
    # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Store frame in history
        self.frame_history.append(gray)
        if len(self.frame_history) > 2:
            self.frame_history.pop(0)
    
    # Calculate motion between frames
        if len(self.frame_history) == 2:
            prev_frame = self.frame_history[0]
            curr_frame = self.frame_history[1]
        
        # Ensure frames have the same dimensions before comparing
            if prev_frame.shape != curr_frame.shape:
                min_height = min(prev_frame.shape[0], curr_frame.shape[0])
                min_width = min(prev_frame.shape[1], curr_frame.shape[1])
            
                if min_height > 0 and min_width > 0:
                    prev_frame = cv2.resize(prev_frame, (min_width, min_height))
                    curr_frame = cv2.resize(curr_frame, (min_width, min_height))
                else:return 0.0
        
        # Calculate absolute difference
            diff = cv2.absdiff(prev_frame, curr_frame)
        
        # Calculate motion score
            motion_score = np.mean(diff) / 255.0
        
            return motion_score
    
        return 0.0
    
    def perform_liveness_test(self, image, face_box, face_landmarks=None, test_type=LivenessTest.BLINK):
        """
        Perform specific liveness test
        
        Args:
            image: Full image
            face_box: Face bounding box (x, y, w, h)
            face_landmarks: Face landmarks (optional)
            test_type: Type of liveness test to perform
            
        Returns:
            (is_live, confidence_score)
        """
        if image is None or face_box is None:
            return False, 0.0
        
        x, y, w, h = face_box
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        if face_region.size == 0:
            return False, 0.0
        
        if test_type == LivenessTest.BLINK:
            # Blink detection test
            if face_landmarks is not None:
                is_blinking = self.detect_blink(face_landmarks)
                return is_blinking, 0.8 if is_blinking else 0.2
            else:
                # Fallback: use eye region analysis
                eye_region = face_region[int(h*0.2):int(h*0.45), int(w*0.25):int(w*0.75)]
                if eye_region.size > 0:
                    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                    eye_variance = np.var(gray_eye)
                    is_live = eye_variance > 50  # Live eyes have more variation
                    confidence = min(eye_variance / 200.0, 1.0)
                    return is_live, confidence
        
        elif test_type == LivenessTest.HEAD_MOVEMENT:
            # Head movement test
            if face_landmarks is not None:
                is_moving = self.detect_head_movement(face_landmarks, image.shape[:2])
                return is_moving, 0.7 if is_moving else 0.3
        
        elif test_type == LivenessTest.MOUTH_MOVEMENT:
            # Mouth movement test
            if face_landmarks is not None:
                is_moving = self.detect_mouth_movement(face_landmarks)
                return is_moving, 0.7 if is_moving else 0.3
        
        elif test_type == LivenessTest.COLOR_SATURATION:
            # Color saturation analysis
            saturation = self.analyze_color_saturation(face_region)
            is_live = saturation > self.color_saturation_threshold
            confidence = saturation
            return is_live, confidence
        
        elif test_type == LivenessTest.TEXTURE_ANALYSIS:
            # Texture analysis
            texture_score = self.analyze_texture(face_region)
            is_live = texture_score > (self.texture_variance_threshold / 10000.0)
            confidence = texture_score
            return is_live, confidence
        
        elif test_type == LivenessTest.MOTION_ANALYSIS:
            # Motion analysis
            motion_score = self.analyze_motion(image, face_box)
            is_live = motion_score > self.motion_threshold / 255.0
            confidence = min(motion_score * 10, 1.0)
            return is_live, confidence
        
        return False, 0.0
    
    def comprehensive_liveness_check(self, image, face_box, face_landmarks=None, 
                                    required_tests=None, min_passing=2):
        """
        Perform comprehensive liveness check using multiple methods
        
        Args:
            image: Full image
            face_box: Face bounding box
            face_landmarks: Face landmarks
            required_tests: List of tests to perform (None = all)
            min_passing: Minimum number of tests that must pass
            
        Returns:
            (is_live, confidence, test_results)
        """
        if required_tests is None:
            required_tests = [
                LivenessTest.COLOR_SATURATION,
                LivenessTest.TEXTURE_ANALYSIS,
                LivenessTest.MOTION_ANALYSIS
            ]
        
        test_results = {}
        passing_tests = 0
        total_confidence = 0.0
        
        for test in required_tests:
            is_live, confidence = self.perform_liveness_test(
                image, face_box, face_landmarks, test
            )
            
            test_results[test.value] = {
                'passed': is_live,
                'confidence': confidence
            }
            
            if is_live:
                passing_tests += 1
            total_confidence += confidence
        
        is_live = passing_tests >= min_passing
        avg_confidence = total_confidence / len(required_tests) if len(required_tests) > 0 else 0.0
        
        return is_live, avg_confidence, test_results
    
    def challenge_response_test(self, image, face_box, challenge="blink"):
        start_time = time.time()
        challenge_completed = False
        response_data = None
        confidence = 0.0
        # This would be called in a loop with multiple frames
        # For now, return a simulated response
        if challenge == "blink":
            # Simulate blink detection
            challenge_completed = True  # Would be set based on actual detection
            confidence = 0.8
        
        elif challenge == "turn_left" or challenge == "turn_right":
            # Simulate head turn detection
            challenge_completed = True
            confidence = 0.7
        
        elif challenge == "smile":
            # Simulate smile detection
            challenge_completed = True
            confidence = 0.75
        
        response_time = time.time() - start_time
        
        return challenge_completed, response_time, confidence


class SimpleFaceDetector:
    def __init__(self):
        self.face_regions = []
        
    def detectMultiScale(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """
        Improved fallback detector using skin color detection and contour analysis
        """
        try:
            # Convert to RGB for better skin detection
            if len(image.shape) == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. Skin color detection (basic HSV-based)
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask for skin color
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # 2. Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # 3. Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            faces = []
            for contour in contours:
                # Filter by size
                area = cv2.contourArea(contour)
                if area < minSize[0] * minSize[1]:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (typical face is roughly 1:1 to 1:1.5)
                aspect_ratio = w / h
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                # Check minimum size
                if w >= minSize[0] and h >= minSize[1]:
                    faces.append([x, y, w, h])
            
            if len(faces) > 0:
                # Merge overlapping boxes
                faces = self._merge_overlapping_boxes(faces)
                return np.array(faces)
            
            # Fallback: If no faces found, use edge-based detection
            return self._edge_based_detection(image, minSize)
            
        except Exception as e:
            print(f"Fallback detector error: {e}")
            # Ultimate fallback: center of image with reasonable size
            h, w = image.shape[:2]
            face_size = min(w, h) // 2
            x = max(0, (w - face_size) // 2)
            y = max(0, (h - face_size) // 2)
            return np.array([[x, y, face_size, face_size]])
    
    def _merge_overlapping_boxes(self, boxes, overlap_threshold=0.3):
        """Merge overlapping bounding boxes"""
        if not boxes:
            return boxes
        
        boxes = sorted(boxes, key=lambda x: x[0])
        merged = []
        
        for box in boxes:
            x1, y1, w1, h1 = box
            x2, y2 = x1 + w1, y1 + h1
            
            merged_box = None
            for i, mbox in enumerate(merged):
                mx1, my1, mw1, mh1 = mbox
                mx2, my2 = mx1 + mw1, my1 + mh1
                
                # Calculate overlap
                inter_x1 = max(x1, mx1)
                inter_y1 = max(y1, my1)
                inter_x2 = min(x2, mx2)
                inter_y2 = min(y2, my2)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    box_area = w1 * h1
                    mbox_area = mw1 * mh1
                    
                    if inter_area / min(box_area, mbox_area) > overlap_threshold:
                        # Merge boxes
                        new_x1 = min(x1, mx1)
                        new_y1 = min(y1, my1)
                        new_x2 = max(x2, mx2)
                        new_y2 = max(y2, my2)
                        merged[i] = [new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1]
                        merged_box = merged[i]
                        break
            
            if merged_box is None:
                merged.append(box)
        
        return merged
    
    def _edge_based_detection(self, image, minSize):
        """Alternative edge-based face detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        faces = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minSize[0] * minSize[1] * 10:  # Larger threshold for edges
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Face-like aspect ratio
            aspect_ratio = w / h
            if 0.7 < aspect_ratio < 1.5:  # More strict for faces
                faces.append([x, y, w, h])
        
        return np.array(faces) if len(faces) > 0 else np.array([])


class FaceDetector:
    def __init__(self, method: str = "hog", min_confidence: float = 0.5,
                 enable_liveness: bool = True,
                 dnn_prototxt_path: str = "deploy.prototxt",
                 dnn_model_path: str = "res10_300x300_ssd_iter_140000.caffemodel",
                 cascade_path: Optional[str] = None):
        """
        Initialize face detector with liveness detection
        
        Args:
            method: Detection method ("dnn", "haar", or "hog")
            min_confidence: Minimum confidence for DNN detection
            enable_liveness: Enable anti-spoofing liveness detection
            dnn_prototxt_path: Path to DNN prototxt file
            dnn_model_path: Path to DNN model file
            cascade_path: Custom path for Haar cascade XML
        """
        self.method = method.lower()
        self.min_confidence = min_confidence
        self.enable_liveness = enable_liveness
        self.dnn_prototxt_path = dnn_prototxt_path
        self.dnn_model_path = dnn_model_path
        self.cascade_path = cascade_path
        
        # Initialize liveness detector if enabled
        if self.enable_liveness:
            self.liveness_detector = LivenessDetector()
            print("✅ Liveness detection (anti-spoofing) enabled")
        
        # Initialize the selected method
        if self.method == "dnn":
            # Try to use DNN-based detector
            try:
                # Check if model files exist
                if not os.path.exists(self.dnn_prototxt_path):
                    print(f"⚠️ {self.dnn_prototxt_path} not found. Download from:")
                    print("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
                    self.method = "haar"
                elif not os.path.exists(self.dnn_model_path):
                    print(f"⚠️ DNN model not found at {self.dnn_model_path}. Download from:")
                    print("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel")
                    self.method = "haar"
                else:
                    self.net = cv2.dnn.readNetFromCaffe(
                        self.dnn_prototxt_path,
                        self.dnn_model_path
                    )
                    self.use_dnn = True
                    print(f"✅ Using DNN-based face detector (model: {os.path.basename(dnn_model_path)})")
            except Exception as e:
                print(f"⚠️ DNN initialization failed: {e}")
                print("Falling back to Haar Cascade")
                self.method = "haar"
                self.use_dnn = False
        
        # Default to Haar Cascade (works without external files)
        if self.method == "haar" or self.method == "hog" or not getattr(self, 'use_dnn', False):
            self.use_dnn = False
            
            # Try to load Haar Cascade with configurable path
            if self.cascade_path and os.path.exists(self.cascade_path):
                try:
                    self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
                    if not self.face_cascade.empty():
                        print(f"✅ Using custom Haar Cascade: {self.cascade_path}")
                    else:
                        raise ValueError("Failed to load cascade")
                except Exception as e:
                    print(f"⚠️ Failed to load custom cascade: {e}")
                    self.cascade_path = None
            
            # If custom path failed or not provided, try defaults
            if not hasattr(self, 'face_cascade') or self.face_cascade is None or self.cascade_path is None:
                cascade_found = False
                
                # Try standard OpenCV path first
                try:
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    if not self.face_cascade.empty():
                        cascade_found = True
                        print("✅ Using Haar Cascade face detector (OpenCV data path)")
                except:
                    pass
                
                # If not found, try current directory
                if not cascade_found:
                    try:
                        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                        if not self.face_cascade.empty():
                            cascade_found = True
                            print("✅ Using Haar Cascade face detector (local file)")
                    except:
                        pass
                
                # If still not found, use improved fallback
                if not cascade_found:
                    print("⚠️ Haar cascade not found. Using improved fallback detector...")
                    self.face_cascade = SimpleFaceDetector()
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        if image is None or image.size == 0:
            return []
        
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            raise ValueError("Input image must be in color (BGR format)")
        
        faces: List[Tuple[int, int, int, int]] = []
        
        if self.use_dnn:
            # DNN-based detection (SSD with ResNet)
            try:
                # Create blob from image
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 
                    1.0, 
                    (300, 300), 
                    (104.0, 177.0, 123.0)
                )
                
                # Perform detection
                self.net.setInput(blob)
                detections = self.net.forward()
                
                # Process detections
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.min_confidence:
                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        x1, y1, x2, y2 = box.astype("int")
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        w = x2 - x1
                        h = y2 - y1
                        
                        # Only add valid faces
                        if w > 20 and h > 20:  # Minimum face size
                            faces.append((x1, y1, w, h))
            except Exception as e:
                print(f"⚠️ Error in DNN face detection: {e}")
                # Fallback to Haar Cascade
                faces = self._detect_faces_haar(image)
        else:
            # Haar Cascade or fallback detection
            faces = self._detect_faces_haar(image)
        
        return faces
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade or fallback"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        try:
            # Detect faces using the loaded cascade or fallback
            faces_detected = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Convert to list of tuples
            faces: List[Tuple[int, int, int, int]] = []
            if len(faces_detected) > 0:
                for (x, y, w, h) in faces_detected:
                    faces.append((int(x), int(y), int(w), int(h)))
            
            return faces
        except Exception as e:
            print(f"⚠️ Error in face detection: {e}")
            return []
    
    def check_liveness(self, image: np.ndarray, face_box: Tuple[int, int, int, int], 
                      face_landmarks: Optional[np.ndarray] = None,
                      test_type: str = "comprehensive") -> Tuple[bool, float, dict]:
        if not self.enable_liveness:
            return True, 1.0, {"liveness_disabled": True}
        
        if image is None or face_box is None:
            return False, 0.0, {"error": "Invalid input"}
        
        try:
            if test_type == "comprehensive":
                # Comprehensive multi-test check
                is_live, confidence, test_results = self.liveness_detector.comprehensive_liveness_check(
                    image, face_box, face_landmarks
                )
                return bool(is_live), float(confidence), test_results
            
            elif test_type == "quick":
                # Quick single test (color saturation + texture)
                test_types = [
                    LivenessTest.COLOR_SATURATION,
                    LivenessTest.TEXTURE_ANALYSIS
                ]
                is_live, confidence, test_results = self.liveness_detector.comprehensive_liveness_check(
                    image, face_box, face_landmarks, test_types, min_passing=1
                )
                return bool(is_live), float(confidence), test_results
            
            elif test_type == "challenge":
                # Interactive challenge-response
                # This would need to be called in a loop
                return True, 0.8, {"challenge_initiated": True}
            
            else:
                # Default to quick test
                return self.check_liveness(image, face_box, face_landmarks, "quick")
                
        except Exception as e:
            print(f"⚠️ Error in liveness detection: {e}")
            return False, 0.0, {"error": str(e)}
    
    def perform_challenge_response(self, challenge_type: str = "blink", 
                                  timeout: int = 10) -> Tuple[bool, float]:
        """
        Perform interactive challenge-response test
        
        Args:
            challenge_type: Type of challenge ('blink', 'turn_left', 'turn_right', 'smile')
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            (challenge_completed, confidence)
        """
        if not self.enable_liveness:
            return True, 1.0
        
        print(f"\n🔄 Please perform the following challenge: {challenge_type.replace('_', ' ').title()}")
        print(f"You have {timeout} seconds to complete it.")
        
        # In a real implementation, this would capture frames and analyze them
        # For now, simulate based on user input
        try:
            if challenge_type == "blink":
                input("Please BLINK naturally, then press Enter...")
                return True, 0.8
            elif challenge_type == "turn_left":
                input("Please turn your head LEFT, then press Enter...")
                return True, 0.7
            elif challenge_type == "turn_right":
                input("Please turn your head RIGHT, then press Enter...")
                return True, 0.7
            elif challenge_type == "smile":
                input("Please SMILE naturally, then press Enter...")
                return True, 0.75
            else:
                return False, 0.0
        except:
            return False, 0.0
    
    def detect_faces_with_confidence(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces with confidence scores
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of tuples containing (bounding_box, confidence)
        """
        if image is None or image.size == 0:
            return []
        
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            raise ValueError("Input image must be in color (BGR format)")
        
        results: List[Tuple[Tuple[int, int, int, int], float]] = []
        
        if self.use_dnn:
            # DNN-based detection with confidence scores
            try:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 
                    1.0, 
                    (300, 300), 
                    (104.0, 177.0, 123.0)
                )
                
                self.net.setInput(blob)
                detections = self.net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence > self.min_confidence:
                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        x1, y1, x2, y2 = box.astype("int")
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        w = x2 - x1
                        h = y2 - y1
                        
                        if w > 20 and h > 20:
                            results.append(((x1, y1, w, h), confidence))
            except Exception as e:
                print(f"⚠️ Error in DNN face detection with confidence: {e}")
                # Fallback to simple detection
                faces = self._detect_faces_haar(image)
                for face in faces:
                    results.append((face, 0.8))  # Default confidence
        else:
            # Simple detection (no confidence scores)
            faces = self._detect_faces_haar(image)
            for face in faces:
                results.append((face, 0.8))  # Default confidence
        
        return results
    
    def extract_face(self, image: np.ndarray, box: Tuple[int, int, int, int], 
                    target_size: Tuple[int, int] = (160, 160),
                    padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract and align face from bounding box
        
        Args:
            image: Original image
            box: Bounding box (x, y, width, height)
            target_size: Output face size
            padding: Percentage to expand bounding box (0.0 to 1.0)
            
        Returns:
            Preprocessed face image or None if extraction fails
        """
        if image is None or image.size == 0:
            return None
        
        x, y, w, h = box
        height, width = image.shape[:2]
        
        # Calculate padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        # Expand bounding box with padding
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        
        # Check if the expanded region is valid
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract face region
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return None
        
        # Resize to target size
        try:
            face = cv2.resize(face_region, target_size)
            
            # Convert to float32 and normalize
            face = face.astype('float32')
            
            # Normalize to [-1, 1] range (common for face recognition models)
            face = (face - 127.5) / 127.5
            
            return face
        except Exception as e:
            print(f"⚠️ Error extracting face: {e}")
            return None
    
    def detect_landmarks(self, face_image: np.ndarray, 
                         face_box: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Detect facial landmarks (eyes, nose, mouth corners)
        
        Args:
            face_image: Face region image
            face_box: Optional bounding box in original image coordinates
            
        Returns:
            Array of landmark points [left_eye, right_eye, nose, left_mouth, right_mouth]
            or None if detection fails
        """
        try:
            # For simplicity, we'll estimate landmarks based on face geometry
            # In production, you'd use dlib or MediaPipe for accurate landmarks
            
            h, w = face_image.shape[:2]
            
            # Estimate landmark positions (5-point model)
            # These are approximate positions based on typical face proportions
            landmarks = np.zeros((5, 2), dtype=np.float32)
            
            # Left eye (approximately 40% from left, 30% from top)
            landmarks[0] = [w * 0.4, h * 0.3]
            
            # Right eye (approximately 60% from left, 30% from top)
            landmarks[1] = [w * 0.6, h * 0.3]
            
            # Nose (center, 50% from top)
            landmarks[2] = [w * 0.5, h * 0.5]
            
            # Left mouth corner (35% from left, 75% from top)
            landmarks[3] = [w * 0.35, h * 0.75]
            
            # Right mouth corner (65% from left, 75% from top)
            landmarks[4] = [w * 0.65, h * 0.75]
            
            # If face_box is provided, convert to original image coordinates
            if face_box is not None:
                x, y, _, _ = face_box
                landmarks[:, 0] += x
                landmarks[:, 1] += y
            
            return landmarks
            
        except Exception as e:
            print(f"⚠️ Error detecting landmarks: {e}")
            return None
    
    def extract_face_with_landmarks(self, image: np.ndarray, box: Tuple[int, int, int, int],
                                   target_size: Tuple[int, int] = (160, 160),
                                   padding: float = 0.2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract face and detect landmarks
        
        Args:
            image: Original image
            box: Bounding box (x, y, width, height)
            target_size: Output face size
            padding: Percentage to expand bounding box
            
        Returns:
            (face_image, landmarks) or (None, None) if extraction fails
        """
        face_image = self.extract_face(image, box, target_size, padding)
        
        if face_image is None:
            return None, None
        
        # Denormalize face image for landmark detection
        if face_image.dtype == np.float32:
            # Convert back to uint8 for landmark detection
            face_uint8 = ((face_image + 1.0) * 127.5).astype(np.uint8)
        else:
            face_uint8 = face_image
        
        landmarks = self.detect_landmarks(face_uint8, box)
        
        return face_image, landmarks
    
    def draw_faces(self, image: np.ndarray, 
                  boxes: List[Tuple[int, int, int, int]],
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Original image
            boxes: List of bounding boxes
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding boxes
        """
        if image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = image.copy()
        
        for (x, y, w, h) in boxes:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        return result
    
    def draw_faces_with_landmarks(self, image: np.ndarray,
                                 boxes: List[Tuple[int, int, int, int]],
                                 landmarks_list: Optional[List[np.ndarray]] = None,
                                 color: Tuple[int, int, int] = (0, 255, 0),
                                 thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes and facial landmarks
        
        Args:
            image: Original image
            boxes: List of bounding boxes
            landmarks_list: List of landmark arrays (optional)
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding boxes and landmarks
        """
        if image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = image.copy()
        
        for i, (x, y, w, h) in enumerate(boxes):
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # Draw landmarks if available
            if landmarks_list and i < len(landmarks_list):
                landmarks = landmarks_list[i]
                if landmarks is not None:
                    # Draw each landmark point
                    for (lx, ly) in landmarks:
                        cv2.circle(result, (int(lx), int(ly)), 3, (0, 0, 255), -1)
                    
                    # Draw connections between landmarks
                    if len(landmarks) >= 5:
                        # Eyes to nose
                        cv2.line(result, 
                                (int(landmarks[0][0]), int(landmarks[0][1])),
                                (int(landmarks[2][0]), int(landmarks[2][1])),
                                (255, 0, 0), 1)
                        cv2.line(result,
                                (int(landmarks[1][0]), int(landmarks[1][1])),
                                (int(landmarks[2][0]), int(landmarks[2][1])),
                                (255, 0, 0), 1)
                        
                        # Mouth
                        cv2.line(result,
                                (int(landmarks[3][0]), int(landmarks[3][1])),
                                (int(landmarks[4][0]), int(landmarks[4][1])),
                                (255, 0, 0), 1)
        
        return result
    
    def draw_faces_with_labels(self, image: np.ndarray,
                              boxes_with_confidences: List[Tuple[Tuple[int, int, int, int], float]],
                              color: Tuple[int, int, int] = (0, 255, 0),
                              thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes with confidence scores
        
        Args:
            image: Original image
            boxes_with_confidences: List of (bounding_box, confidence)
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding boxes and confidence scores
        """
        if image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = image.copy()
        
        for (x, y, w, h), confidence in boxes_with_confidences:
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            
            # Add confidence text
            text = f"{confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            
            # Calculate text size
            text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
            
            # Draw background for text
            cv2.rectangle(result, 
                         (x, y - text_size[1] - 5), 
                         (x + text_size[0], y), 
                         color, 
                         -1)
            
            # Draw text
            cv2.putText(result, text, 
                       (x, y - 5), 
                       font, font_scale, 
                       (0, 0, 0), text_thickness)
        
        return result
    
    def draw_liveness_result(self, image: np.ndarray, face_box: Tuple[int, int, int, int],
                           is_live: bool, confidence: float, test_results: dict) -> np.ndarray:
        """
        Draw liveness detection results on image
        
        Args:
            image: Original image
            face_box: Face bounding box
            is_live: Liveness result
            confidence: Confidence score
            test_results: Detailed test results
            
        Returns:
            Image with liveness information
        """
        if image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = image.copy()
        x, y, w, h = face_box
        
        # Draw face box with liveness color
        color = (0, 255, 0) if is_live else (0, 0, 255)  # Green for live, red for spoof
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # Draw liveness status
        status_text = "LIVE" if is_live else "SPOOF"
        status_color = (0, 255, 0) if is_live else (0, 0, 255)
        
        cv2.putText(result, f"Status: {status_text}", (x, y - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(result, f"Confidence: {confidence:.2f}", (x, y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw test results summary
        if test_results and len(test_results) > 0:
            y_offset = y + h + 20
            for test_name, test_result in test_results.items():
                if isinstance(test_result, dict) and 'passed' in test_result:
                    test_status = "✓" if test_result['passed'] else "✗"
                    test_color = (0, 255, 0) if test_result['passed'] else (0, 0, 255)
                    test_conf = test_result.get('confidence', 0)
                    
                    text = f"{test_status} {test_name}: {test_conf:.2f}"
                    cv2.putText(result, text, (x, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, test_color, 1)
                    y_offset += 15
        
        return result