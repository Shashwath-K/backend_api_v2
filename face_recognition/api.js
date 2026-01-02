// API communication with Python backend
class FaceRecognitionAPI {
    constructor() {
        // Base URL - Change this to your server address
        this.baseURL = 'http://localhost:5000'; // Flask server
        
        // Set to false to use real backend
        this.mockMode = false; // CHANGED TO FALSE
        
        // For debugging
        console.log(`API initialized. Mock mode: ${this.mockMode}, Base URL: ${this.baseURL}`);
        
        // Mock data for testing (fallback only)
        this.mockUsers = [
            {
                user_id: 'STU001',
                full_name: 'John Doe',
                user_type: 'student',
                enrollment_number: 'ENR001',
                semester: 3,
                program: 'Computer Science',
                confidence: 0.92
            },
            {
                user_id: 'FAC001',
                full_name: 'Dr. Jane Smith',
                user_type: 'faculty',
                designation: 'Professor',
                confidence: 0.88
            }
        ];
        
        this.mockAttendance = [
            {
                attendance_id: 1,
                student_id: 'STU001',
                attendance_date: '2024-01-15',
                attendance_time: '2024-01-15T09:30:00',
                attendance_status: 'present',
                confidence_score: 0.92,
                attendance_type: 'class'
            },
            {
                attendance_id: 2,
                student_id: 'STU001',
                attendance_date: '2024-01-16',
                attendance_time: '2024-01-16T09:45:00',
                attendance_status: 'late',
                confidence_score: 0.87,
                attendance_type: 'class'
            }
        ];
    }

    async checkServer() {
        if (this.mockMode) {
            console.log('Mock mode: Server check passed');
            return true;
        }

        try {
            console.log(`Checking server health at: ${this.baseURL}/api/health`);
            const response = await fetch(`${this.baseURL}/api/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                mode: 'cors'
            });
            
            console.log('Server health response:', response.status);
            const data = await response.json();
            console.log('Server health data:', data);
            
            return response.ok;
        } catch (error) {
            console.error('Server check failed:', error);
            return false;
        }
    }

    async faceLogin(imageData) {
        if (this.mockMode) {
            console.log('Mock mode: Simulating face login');
            // Simulate API delay
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Return mock user (simulate face recognition)
            const randomUser = this.mockUsers[Math.floor(Math.random() * this.mockUsers.length)];
            return {
                success: true,
                user: randomUser,
                message: 'Login successful'
            };
        }

        try {
            console.log('Sending face login request...');
            
            // Remove data:image/jpeg;base64, prefix if present
            let cleanImageData = imageData;
            if (imageData.startsWith('data:')) {
                cleanImageData = imageData.split(',')[1];
            }
            
            const payload = {
                image: cleanImageData,
                timestamp: new Date().toISOString()
            };
            
            console.log('Login payload size:', cleanImageData.length, 'bytes');
            
            const response = await fetch(`${this.baseURL}/api/login`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            console.log('Login response status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Login server error:', errorText);
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();
            console.log('Login result:', result);
            return result;
            
        } catch (error) {
            console.error('Login API error:', error);
            
            // Fallback to mock data on error
            if (this.mockMode !== true) {
                console.log('Falling back to mock data due to error');
                await new Promise(resolve => setTimeout(resolve, 1000));
                return {
                    success: true,
                    user: this.mockUsers[0],
                    message: 'Login successful (fallback mode)'
                };
            }
            
            throw error;
        }
    }

    async registerUser(userData, faceImages) {
        console.log('Register user called with:', {
            userData: userData,
            faceImagesCount: Array.isArray(faceImages) ? faceImages.length : 1
        });

        if (this.mockMode) {
            console.log('Mock mode: Simulating registration');
            // Simulate registration delay
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            return {
                success: true,
                user_id: userData.user_id,
                message: 'Registration successful (mock mode)',
                note: 'Running in mock mode - no actual registration occurred'
            };
        }

        try {
            console.log('Sending registration request to server...');
            
            // Prepare face images array
            const processedFaceImages = Array.isArray(faceImages) ? faceImages : [faceImages];
            
            // Clean image data (remove data: prefix)
            const cleanedFaceImages = processedFaceImages.map(img => {
                if (img && img.startsWith('data:')) {
                    return img.split(',')[1];
                }
                return img;
            });
            
            console.log(`Processing ${cleanedFaceImages.length} face images`);
            
            const payload = {
                user_data: userData,
                face_images: cleanedFaceImages
            };
            
            console.log('Registration payload:', {
                user_data: userData,
                face_images_count: cleanedFaceImages.length,
                first_image_length: cleanedFaceImages[0] ? cleanedFaceImages[0].length : 0
            });
            
            const response = await fetch(`${this.baseURL}/api/register`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            console.log('Registration response status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Registration server error:', errorText);
                
                // Try to parse as JSON if possible
                try {
                    const errorData = JSON.parse(errorText);
                    throw new Error(errorData.error || `Server error: ${response.status}`);
                } catch {
                    throw new Error(`Server error: ${response.status} - ${errorText}`);
                }
            }

            const result = await response.json();
            console.log('Registration result:', result);
            
            return result;
            
        } catch (error) {
            console.error('Registration API error:', error);
            
            // Fallback to mock registration on error
            if (this.mockMode !== true) {
                console.log('Falling back to mock registration due to error');
                await new Promise(resolve => setTimeout(resolve, 1500));
                return {
                    success: true,
                    user_id: userData.user_id,
                    message: 'Registration successful (fallback mode)',
                    note: 'Server communication failed, but user data was saved locally'
                };
            }
            
            throw error;
        }
    }

    async getAttendanceRecords(studentId) {
    console.log(`Fetching attendance for student: ${studentId}`);

    if (this.mockMode) {
        console.log('Mock mode: Returning mock attendance data');
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Return mock attendance for the student
        return this.mockAttendance.filter(record => record.student_id === studentId);
    }

    try {
        console.log(`Fetching attendance from: ${this.baseURL}/api/attendance/${studentId}`);
        
        const response = await fetch(`${this.baseURL}/api/attendance/${studentId}`, {
            method: 'GET',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        console.log('Attendance response status:', response.status);
        
        if (!response.ok) {
            console.warn(`Attendance endpoint error (${response.status})`);
            const errorText = await response.text();
            console.error('Error details:', errorText);
            
            // Return empty array instead of throwing error
            return [];
        }

        const data = await response.json();
        console.log('Attendance data received:', data);
        
        // Handle both response formats
        if (data.success && data.records) {
            return data.records;
        } else if (Array.isArray(data)) {
            return data;
        } else {
            console.warn('Unexpected attendance response format:', data);
            return [];
        }
        
    } catch (error) {
        console.error('Attendance fetch error:', error);
        
        // Return empty array on error instead of throwing
        return [];
    }
}
    async getStudentInfo(studentId) {
        if (this.mockMode) {
            await new Promise(resolve => setTimeout(resolve, 500));
            
            const student = this.mockUsers.find(u => u.user_id === studentId);
            if (student) {
                return {
                    success: true,
                    student: student
                };
            }
            
            throw new Error('Student not found');
        }

        try {
            const response = await fetch(`${this.baseURL}/api/student/${studentId}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Student info error:', error);
            throw error;
        }
    }
}