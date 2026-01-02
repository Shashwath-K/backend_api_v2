import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any
from numpy.typing import NDArray

class SmartFaceValidator:
    def __init__(self, auto_adjust: bool = True):
        """
        Smart validator that can auto-adjust thresholds based on camera conditions
        
        Args:
            auto_adjust: Whether to auto-adjust thresholds based on camera test
        """
        self.auto_adjust = auto_adjust
        self.calibrated = False
        
        # Default thresholds (strict)
        self.default_thresholds = {
            'blur_threshold': 100.0,
            'min_brightness': 0.2,
            'max_brightness': 0.8,
            'min_contrast': 20.0,
            'pose_symmetry_threshold': 0.85,
            'pose_aspect_ratio_threshold': 0.7
        }
        
        # Current thresholds (will be adjusted)
        self.current_thresholds = self.default_thresholds.copy()
        
        # Camera calibration data
        self.camera_stats = {}
        
    def calibrate_camera(self, camera_index: int = 0, test_frames: int = 30):
        """
        Calibrate thresholds based on camera conditions
        
        Args:
            camera_index: Camera device index
            test_frames: Number of frames to analyze
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")
        
        # Set decent resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("🔍 Calibrating camera thresholds...")
        print(f"Analyzing {test_frames} frames")
        
        blur_scores = []
        brightness_scores = []
        contrast_scores = []
        
        frame_count = 0
        while frame_count < test_frames:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            # 1. Blur score (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = float(laplacian.var())
            blur_scores.append(blur_score)
            
            # 2. Brightness score
            brightness_score = np.mean(gray) / 255.0
            brightness_scores.append(brightness_score)
            
            # 3. Contrast score
            contrast_score = gray.std()
            contrast_scores.append(contrast_score)
            
            frame_count += 1
            
            # Show progress
            if frame_count % 10 == 0:
                print(f"  Processed {frame_count}/{test_frames} frames")
        
        cap.release()
        
        # Calculate statistics
        self.camera_stats = {
            'blur_mean': np.mean(blur_scores),
            'blur_std': np.std(blur_scores),
            'blur_min': np.min(blur_scores),
            'blur_max': np.max(blur_scores),
            'brightness_mean': np.mean(brightness_scores),
            'brightness_std': np.std(brightness_scores),
            'brightness_min': np.min(brightness_scores),
            'brightness_max': np.max(brightness_scores),
            'contrast_mean': np.mean(contrast_scores),
            'contrast_std': np.std(contrast_scores),
            'contrast_min': np.min(contrast_scores),
            'contrast_max': np.max(contrast_scores),
        }
        
        # Adjust thresholds based on camera stats
        if self.auto_adjust:
            self._adjust_thresholds()
        
        self.calibrated = True
        
        return self.camera_stats
    
    def _adjust_thresholds(self):
        """Adjust thresholds based on camera statistics"""
        stats = self.camera_stats
        
        # Adjust blur threshold (use 25th percentile as threshold)
        blur_threshold = max(30.0, stats['blur_mean'] - stats['blur_std'])
        
        # Adjust brightness thresholds (center around mean)
        brightness_center = stats['brightness_mean']
        brightness_range = 0.3  # ±0.15 from center
        min_brightness = max(0.1, brightness_center - brightness_range/2)
        max_brightness = min(0.9, brightness_center + brightness_range/2)
        
        # Adjust contrast threshold (use 25th percentile)
        contrast_threshold = max(10.0, stats['contrast_mean'] - stats['contrast_std'])
        
        # Update thresholds
        self.current_thresholds = {
            'blur_threshold': blur_threshold,
            'min_brightness': min_brightness,
            'max_brightness': max_brightness,
            'min_contrast': contrast_threshold,
            'pose_symmetry_threshold': 0.7,  # Keep these reasonable
            'pose_aspect_ratio_threshold': 0.6,
        }
    
    def get_threshold_summary(self) -> str:
        """Get formatted summary of thresholds"""
        summary = []
        summary.append("🔧 CURRENT VALIDATION THRESHOLDS:")
        summary.append("-" * 45)
        
        for key, value in self.current_thresholds.items():
            if 'brightness' in key or 'pose' in key:
                summary.append(f"{key:<30}: {value:.3f}")
            else:
                summary.append(f"{key:<30}: {value:.1f}")
        
        if self.camera_stats:
            summary.append("\n📊 CAMERA STATISTICS:")
            summary.append("-" * 45)
            stats_display = {
                'blur_mean': 'Blur (mean)',
                'brightness_mean': 'Brightness (mean)',
                'contrast_mean': 'Contrast (mean)',
            }
            
            for stat_key, display_name in stats_display.items():
                value = self.camera_stats.get(stat_key, 0)
                if 'brightness' in stat_key:
                    summary.append(f"{display_name:<30}: {value:.3f}")
                else:
                    summary.append(f"{display_name:<30}: {value:.1f}")
        
        return "\n".join(summary)
    
    # Validation methods (using current thresholds)
    def validate_blur(self, face_image: NDArray[np.uint8]) -> Tuple[bool, float]:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())
        
        is_clear = laplacian_var > self.current_thresholds['blur_threshold']
        return is_clear, laplacian_var
    
    def validate_brightness(self, face_image: NDArray[np.uint8]) -> Tuple[bool, float]:
        if face_image.dtype != np.uint8:
            face_image = face_image.astype(np.uint8)
        
        if len(face_image.shape) == 3:
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            brightness_value = float(np.mean(hsv[:, :, 2]))
        else:
            brightness_value = float(np.mean(face_image))
        
        brightness_score = brightness_value / 255.0
        min_b = self.current_thresholds['min_brightness']
        max_b = self.current_thresholds['max_brightness']
        
        is_proper = min_b <= brightness_score <= max_b
        return is_proper, brightness_score
    
    def validate_contrast(self, face_image: NDArray[np.uint8]) -> Tuple[bool, float]:
        if face_image.dtype != np.uint8:
            face_image = face_image.astype(np.uint8)
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        contrast_value = float(gray.std())
        
        contrast_score = contrast_value / 128.0
        has_good_contrast = contrast_value > self.current_thresholds['min_contrast']
        
        return has_good_contrast, contrast_score
    
    def validate_face_quality(self, face_image: NDArray[np.uint8],
                            face_box: Tuple[int, int, int, int],
                            original_image_shape: Optional[Tuple[int, int]] = None,
                            debug: bool = False) -> Dict[str, Any]:
        """Full validation using calibrated thresholds"""
        
        if debug and not self.calibrated:
            print("⚠️ Warning: Validator not calibrated. Using default thresholds.")
        
        results = {
            'is_valid': True,
            'issues': [],
            'scores': {},
            'warnings': [],
            'thresholds_used': self.current_thresholds.copy()
        }
        
        # Run validations
        try:
            # Blur
            is_clear, blur_score = self.validate_blur(face_image)
            results['scores']['blur'] = blur_score
            if not is_clear:
                results['is_valid'] = False
                results['issues'].append('blurry')
        
            # Brightness
            is_proper, brightness_score = self.validate_brightness(face_image)
            results['scores']['brightness'] = brightness_score
            if not is_proper:
                results['is_valid'] = False
                results['issues'].append('poor_lighting')
        
            # Contrast
            has_good_contrast, contrast_score = self.validate_contrast(face_image)
            results['scores']['contrast'] = contrast_score
            if not has_good_contrast:
                results['warnings'].append('low_contrast')
        
        except Exception as e:
            if debug:
                print(f"Validation error: {e}")
        
        return results

# Test function
def test_smart_validator():
    """Test the smart validator with camera calibration"""
    print("=" * 70)
    print("🤖 SMART FACE VALIDATOR TEST")
    print("=" * 70)
    
    # Create validator with auto-adjust
    validator = SmartFaceValidator(auto_adjust=True)
    
    # Calibrate camera
    try:
        stats = validator.calibrate_camera(camera_index=0, test_frames=20)
        
        # Show results
        print("\n" + validator.get_threshold_summary())
        
        # Test with camera feed
        print("\n🎯 TESTING VALIDATION WITH CALIBRATED THRESHOLDS")
        print("Press 'q' to quit, 'c' to capture and validate")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            display = frame.copy()
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                # Validate using calibrated thresholds
                results = validator.validate_face_quality(
                    face_roi, 
                    (x, y, w, h), 
                    frame.shape[:2],
                    debug=False
                )
                
                # Draw results
                color = (0, 255, 0) if results['is_valid'] else (0, 0, 255)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                
                # Display scores
                status = "PASS" if results['is_valid'] else "FAIL"
                cv2.putText(display, f"Status: {status}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display thresholds info
                cv2.putText(display, f"Blur: {results['scores'].get('blur', 0):.1f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display, f"Bright: {results['scores'].get('brightness', 0):.3f}", 
                          (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Smart Validator Test', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and len(faces) > 0:
                # Capture and show detailed validation
                results = validator.validate_face_quality(
                    face_roi, 
                    (x, y, w, h), 
                    frame.shape[:2],
                    debug=True
                )
                
                print(f"\n📸 CAPTURE VALIDATION RESULTS:")
                print(f"  Blur: {results['scores'].get('blur', 0):.1f} (threshold: {results['thresholds_used']['blur_threshold']:.1f})")
                print(f"  Brightness: {results['scores'].get('brightness', 0):.3f} (range: [{results['thresholds_used']['min_brightness']:.3f}, {results['thresholds_used']['max_brightness']:.3f}])")
                print(f"  Contrast: {results['scores'].get('contrast', 0):.3f} (threshold: {results['thresholds_used']['min_contrast']:.1f})")
                print(f"  Overall: {'PASS' if results['is_valid'] else 'FAIL'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_smart_validator()