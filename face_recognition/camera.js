// Camera handling for mobile devices
class CameraManager {
    constructor() {
        this.videoElement = document.getElementById('camera-preview');
        this.canvasElement = document.getElementById('face-canvas');
        this.ctx = this.canvasElement?.getContext('2d');
        this.stream = null;
        this.isCameraActive = false;
        this.cameraMode = 'environment'; // 'user' for front, 'environment' for back
    }

    async startCamera() {
        try {
            // Check if camera is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera not supported on this device');
            }

            // Stop existing stream if any
            if (this.stream) {
                this.stopCamera();
            }

            // Get camera constraints for mobile
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: this.cameraMode,
                    frameRate: { ideal: 30 }
                },
                audio: false
            };

            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Set video source
            this.videoElement.srcObject = this.stream;
            this.isCameraActive = true;

            // Wait for video to be ready
            await new Promise((resolve) => {
                this.videoElement.onloadedmetadata = () => {
                    this.videoElement.play();
                    
                    // Set canvas dimensions to match video
                    this.canvasElement.width = this.videoElement.videoWidth;
                    this.canvasElement.height = this.videoElement.videoHeight;
                    
                    resolve();
                };
            });

            // Enable capture button
            const captureBtn = document.getElementById('capture-btn');
            if (captureBtn) {
                captureBtn.disabled = false;
                captureBtn.textContent = '⚡ Capture & Login';
            }

            // Update start button
            const startBtn = document.getElementById('start-camera-btn');
            if (startBtn) {
                startBtn.textContent = '🔄 Restart Camera';
                startBtn.className = 'btn btn-warning';
            }

            console.log('Camera started successfully');
            return true;

        } catch (error) {
            console.error('Camera error:', error);
            
            let errorMessage = 'Camera access failed: ';
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Please allow camera access in your browser settings';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No camera found on this device';
            } else if (error.name === 'NotSupportedError') {
                errorMessage += 'Camera not supported on this browser';
            } else {
                errorMessage += error.message;
            }

            // Show error to user
            if (window.app) {
                window.app.showToast(errorMessage, 'error');
            }
            
            return false;
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
        
        this.isCameraActive = false;

        // Disable capture button
        const captureBtn = document.getElementById('capture-btn');
        if (captureBtn) {
            captureBtn.disabled = true;
            captureBtn.textContent = 'Start Camera First';
        }

        // Reset start button
        const startBtn = document.getElementById('start-camera-btn');
        if (startBtn) {
            startBtn.textContent = '📷 Start Camera';
            startBtn.className = 'btn btn-success';
        }
    }

    toggleCamera() {
        this.cameraMode = this.cameraMode === 'user' ? 'environment' : 'user';
        this.stopCamera();
        setTimeout(() => this.startCamera(), 500);
    }

    async captureImage() {
        if (!this.isCameraActive || !this.videoElement) {
            throw new Error('Camera not active');
        }

        // Draw current video frame to canvas
        this.ctx.drawImage(
            this.videoElement,
            0, 0,
            this.canvasElement.width,
            this.canvasElement.height
        );

        // Get image data as base64
        const imageData = this.canvasElement.toDataURL('image/jpeg', 0.8);
        
        // Draw face detection rectangle (simulated)
        this.drawFaceRectangle();
        
        return imageData;
    }

    drawFaceRectangle() {
        // Simulate face detection by drawing a rectangle
        // In production, you would use a face detection library
        
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        
        // Center rectangle (simulated face)
        const width = this.canvasElement.width;
        const height = this.canvasElement.height;
        const faceWidth = width * 0.4;
        const faceHeight = height * 0.5;
        const faceX = (width - faceWidth) / 2;
        const faceY = (height - faceHeight) / 2;
        
        this.ctx.rect(faceX, faceY, faceWidth, faceHeight);
        this.ctx.stroke();
        
        // Add text
        this.ctx.fillStyle = '#00ff00';
        this.ctx.font = '16px Arial';
        this.ctx.fillText('Face Detected', faceX, faceY - 10);
    }

    setupCamera(pageType) {
        // Reset camera state
        this.stopCamera();
        
        // Setup based on page type
        switch(pageType) {
            case 'login':
                this.cameraMode = 'user'; // Front camera for login
                break;
            case 'register':
                this.cameraMode = 'environment'; // Back camera for registration (if available)
                break;
        }
        
        // Auto-start camera after a delay
        setTimeout(() => {
            if (this.isUserOnMobile()) {
                this.startCamera();
            }
        }, 1000);
    }

    isUserOnMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }
}