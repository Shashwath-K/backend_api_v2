// Main Application Logic - FIXED VERSION
class FaceRecognitionApp {
    constructor() {
        this.currentPage = 'home';
        this.currentUser = null;
        this.isLoading = false; // Track loading state
        this.api = new FaceRecognitionAPI();
        this.camera = new CameraManager();
        this.init();
    }

    init() {
        // Setup navigation
        this.setupNavigation();
        
        // Setup forms
        this.setupForms();
        
        // Check server connection
        this.checkConnection();
        
        // Setup event listeners
        this.setupEventListeners();
        
        console.log('✅ App initialized');
    }

    setupNavigation() {
        // Handle navigation clicks
        document.querySelectorAll('[data-page]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.getAttribute('data-page');
                this.navigateTo(page);
            });
        });

        // Handle browser back/forward
        window.addEventListener('hashchange', () => {
            const page = window.location.hash.substring(1) || 'home';
            this.navigateTo(page);
        });

        // Initial navigation
        const initialPage = window.location.hash.substring(1) || 'home';
        this.navigateTo(initialPage);
    }

    navigateTo(page) {
        // Ensure loading is hidden when navigating
        this.hideLoading();
        
        // Hide all pages
        document.querySelectorAll('.page').forEach(p => {
            p.style.display = 'none';
            p.classList.remove('active');
        });

        // Show target page
        const targetPage = document.getElementById(`${page}-page`);
        if (targetPage) {
            targetPage.style.display = 'block';
            targetPage.classList.add('active');
            this.currentPage = page;
            
            // Update active nav link
            document.querySelectorAll('.nav-link').forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('data-page') === page) {
                    link.classList.add('active');
                }
            });

            // Update URL hash
            window.location.hash = page;

            // Page-specific setup
            this.handlePageLoad(page);
        }
    }

    handlePageLoad(page) {
        switch(page) {
            case 'login':
                this.setupLoginPage();
                break;
            case 'register':
                this.setupRegistrationForm();
                break;
            case 'attendance':
                this.setupAttendancePage();
                break;
        }
    }

    setupLoginPage() {
        // Check if we're in registration mode
        const regData = localStorage.getItem('registration_data');
        
        if (regData) {
            // Registration mode - update UI
            const header = document.querySelector('#login-page .card-header');
            const title = document.querySelector('#login-page .card-header h4');
            const captureBtn = document.querySelector('#capture-btn');
            
            if (header && title && captureBtn) {
                title.textContent = '📸 Face Capture for Registration';
                header.className = 'card-header bg-success text-white';
                captureBtn.innerHTML = '⚡ Capture & Register';
            }
        } else {
            // Normal login mode - ensure UI is reset
            const header = document.querySelector('#login-page .card-header');
            const title = document.querySelector('#login-page .card-header h4');
            const captureBtn = document.querySelector('#capture-btn');
            
            if (header && title && captureBtn) {
                title.textContent = '🔐 Face Login';
                header.className = 'card-header bg-primary text-white';
                captureBtn.innerHTML = '⚡ Capture & Login';
            }
        }
        
        // Setup camera
        this.camera.setupCamera('login');
        
        // Setup event listeners for this page
        this.setupLoginEventListeners();
    }

    setupLoginEventListeners() {
        // Start camera button
        const startCameraBtn = document.getElementById('start-camera-btn');
        if (startCameraBtn) {
            // Remove existing listeners
            const newStartBtn = startCameraBtn.cloneNode(true);
            startCameraBtn.parentNode.replaceChild(newStartBtn, startCameraBtn);
            
            // Add new listener
            newStartBtn.addEventListener('click', () => {
                this.camera.startCamera();
            });
        }
        
        // Capture button
        const captureBtn = document.getElementById('capture-btn');
        if (captureBtn) {
            // Remove existing listeners
            const newCaptureBtn = captureBtn.cloneNode(true);
            captureBtn.parentNode.replaceChild(newCaptureBtn, captureBtn);
            
            // Add new listener
            newCaptureBtn.addEventListener('click', async () => {
                await this.handleFaceCapture();
            });
        }
    }

    setupForms() {
        // Registration form
        const regForm = document.getElementById('registration-form');
        if (regForm) {
            regForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.handleRegistration();
            });
        }

        // User type change - show/hide student fields
        const userTypeSelect = document.getElementById('user-type');
        if (userTypeSelect) {
            userTypeSelect.addEventListener('change', (e) => {
                this.toggleStudentFields(e.target.value === 'student');
            });
            
            // Initialize on page load
            this.toggleStudentFields(userTypeSelect.value === 'student');
        }
    }

    toggleStudentFields(show) {
        const studentFields = document.getElementById('student-fields');
        if (studentFields) {
            studentFields.style.display = show ? 'block' : 'none';
            
            // Toggle required attribute
            const studentInputs = studentFields.querySelectorAll('input');
            studentInputs.forEach(input => {
                input.required = show;
            });
        }
    }

    async checkConnection() {
        const statusElement = document.getElementById('connection-status');
        if (!statusElement) return;

        try {
            this.showLoading('Checking server connection...');
            const isConnected = await this.api.checkServer();
            
            if (isConnected) {
                statusElement.innerHTML = '✅ Connected to server';
                statusElement.className = 'alert alert-success';
            } else {
                statusElement.innerHTML = '❌ Server connection failed';
                statusElement.className = 'alert alert-danger';
            }
        } catch (error) {
            statusElement.innerHTML = `❌ Connection error: ${error.message}`;
            statusElement.className = 'alert alert-danger';
        } finally {
            this.hideLoading();
        }
    }

    async handleRegistration() {
        try {
            this.showLoading('Processing registration...');
            
            const formData = {
                full_name: document.getElementById('full-name').value.trim(),
                user_type: document.getElementById('user-type').value,
                user_id: document.getElementById('user-id').value.trim(),
                email: document.getElementById('email').value.trim()
            };

            // Add student-specific fields if applicable
            if (formData.user_type === 'student') {
                formData.enrollment_number = document.getElementById('enrollment-number').value.trim();
                formData.semester = document.getElementById('semester').value;
                formData.program = document.getElementById('program').value.trim();
                formData.batch_year = document.getElementById('batch-year').value.trim();
            }

            // Validate data
            if (!this.validateRegistrationData(formData)) {
                return;
            }

            // Format data for server
            const serverData = {
                full_name: formData.full_name,
                user_type: formData.user_type,
                user_id: formData.user_id,
                email: formData.email
            };

            // Add student data if student
            if (formData.user_type === 'student') {
                serverData.student = {
                    student_id: formData.user_id,
                    enrollment_number: formData.enrollment_number,
                    semester: formData.semester,
                    program: formData.program,
                    batch_year: formData.batch_year,
                    full_name: formData.full_name,
                    user_id: formData.user_id
                };
            }

            // Store data for face capture
            localStorage.setItem('registration_data', JSON.stringify(serverData));
            
            // Navigate to face capture
            this.navigateTo('login');
            
            this.showToast('Please capture your face for registration', 'info');
            
        } catch (error) {
            this.showToast(`Registration error: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    validateRegistrationData(data) {
        // Basic validation
        if (!data.full_name || data.full_name.length < 2) {
            this.showToast('Full name is required (minimum 2 characters)', 'error');
            return false;
        }

        if (!data.user_id || data.user_id.length < 3) {
            this.showToast('User ID is required (minimum 3 characters)', 'error');
            return false;
        }

        if (data.user_type === 'student') {
            if (!data.enrollment_number || data.enrollment_number.trim() === '') {
                this.showToast('Enrollment number is required for students', 'error');
                return false;
            }
            
            if (!data.program || data.program.trim() === '') {
                this.showToast('Program is required for students', 'error');
                return false;
            }
        }

        return true;
    }

    setupRegistrationForm() {
        // Ensure student fields are properly initialized
        const userTypeSelect = document.getElementById('user-type');
        if (userTypeSelect) {
            this.toggleStudentFields(userTypeSelect.value === 'student');
        }
    }

    setupAttendancePage() {
        const fetchBtn = document.getElementById('fetch-attendance-btn');
        if (fetchBtn) {
            // Remove existing listeners
            const newFetchBtn = fetchBtn.cloneNode(true);
            fetchBtn.parentNode.replaceChild(newFetchBtn, fetchBtn);
            
            newFetchBtn.addEventListener('click', async () => {
                await this.fetchAttendance();
            });
        }

        // Also allow pressing Enter in the input field
        const studentIdInput = document.getElementById('attendance-student-id');
        if (studentIdInput) {
            studentIdInput.addEventListener('keypress', async (e) => {
                if (e.key === 'Enter') {
                    await this.fetchAttendance();
                }
            });
        }

        // Auto-fetch if we have a current user
        if (this.currentUser && this.currentUser.user_type === 'student') {
            document.getElementById('attendance-student-id').value = this.currentUser.user_id;
            setTimeout(async () => {
                await this.fetchAttendance();
            }, 500);
        }
    }

    async fetchAttendance() {
        // Prevent multiple simultaneous requests
        if (this.isLoading) {
            console.log('⚠️ Request already in progress');
            return;
        }

        const studentId = document.getElementById('attendance-student-id').value.trim();
        if (!studentId) {
            this.showToast('Please enter a Student ID', 'error');
            return;
        }

        this.isLoading = true;
        this.showLoading('Fetching attendance records...');

        try {
            console.log(`📋 Fetching attendance for: ${studentId}`);
            const records = await this.api.getAttendanceRecords(studentId);
            console.log('✅ Attendance records received:', records);
            
            this.displayAttendanceRecords(records);
            this.calculateAttendanceStats(records);
            
        } catch (error) {
            console.error('❌ Error fetching attendance:', error);
            this.showToast(`Error fetching attendance: ${error.message}`, 'error');
            
            // Show empty results on error
            this.displayAttendanceRecords([]);
            document.getElementById('attendance-stats').style.display = 'none';
            
        } finally {
            this.isLoading = false;
            this.hideLoading();
        }
    }

    displayAttendanceRecords(records) {
        const resultsContainer = document.getElementById('attendance-results');
        
        if (!resultsContainer) {
            console.error('❌ attendance-results container not found');
            return;
        }
        
        if (!records || records.length === 0) {
            resultsContainer.innerHTML = `
                <div class="alert alert-warning">
                    📭 No attendance records found for this student
                </div>
            `;
            if (document.getElementById('attendance-stats')) {
                document.getElementById('attendance-stats').style.display = 'none';
            }
            return;
        }

        // Check if records is an array
        let recordsArray = records;
        if (!Array.isArray(records)) {
            if (records.records && Array.isArray(records.records)) {
                recordsArray = records.records;
            } else if (records.data && Array.isArray(records.data)) {
                recordsArray = records.data;
            } else {
                resultsContainer.innerHTML = `
                    <div class="alert alert-warning">
                        📭 Invalid records format received
                    </div>
                `;
                return;
            }
        }

        let html = `
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Time</th>
                            <th>Status</th>
                            <th>Confidence</th>
                            <th>Type</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        recordsArray.forEach(record => {
            // Handle different record formats
            const attendanceDate = record.attendance_date || record.date || record.timestamp;
            const attendanceTime = record.attendance_time || record.time;
            const status = record.attendance_status || record.status || 'unknown';
            const confidence = record.confidence_score || record.confidence || 0;
            const type = record.attendance_type || record.type || 'Class';
            
            // Parse date/time
            let dateStr, timeStr;
            try {
                const dateObj = new Date(attendanceDate);
                dateStr = dateObj.toLocaleDateString();
                timeStr = attendanceTime || dateObj.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            } catch (e) {
                dateStr = attendanceDate || 'N/A';
                timeStr = attendanceTime || 'N/A';
            }
            
            const statusClass = {
                'present': 'badge bg-success',
                'absent': 'badge bg-danger',
                'late': 'badge bg-warning',
                'present (late)': 'badge bg-warning'
            }[status.toLowerCase()] || 'badge bg-secondary';

            html += `
                <tr>
                    <td>${dateStr}</td>
                    <td>${timeStr}</td>
                    <td><span class="${statusClass}">${status}</span></td>
                    <td>${confidence ? (confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
                    <td>${type}</td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;

        resultsContainer.innerHTML = html;
        
        const statsElement = document.getElementById('attendance-stats');
        if (statsElement) {
            statsElement.style.display = 'block';
        }
    }

    calculateAttendanceStats(records) {
        let recordsArray = records;
        
        // Handle different record formats
        if (!Array.isArray(records)) {
            if (records.records && Array.isArray(records.records)) {
                recordsArray = records.records;
            } else if (records.data && Array.isArray(records.data)) {
                recordsArray = records.data;
            } else {
                recordsArray = [];
            }
        }

        const stats = {
            present: 0,
            absent: 0,
            late: 0,
            total: recordsArray.length
        };

        recordsArray.forEach(record => {
            const status = (record.attendance_status || record.status || '').toLowerCase();
            
            if (status.includes('present') && status.includes('late')) {
                stats.late++;
            } else if (status.includes('present')) {
                stats.present++;
            } else if (status.includes('absent')) {
                stats.absent++;
            } else if (status.includes('late')) {
                stats.late++;
            }
        });

        // Update UI if elements exist
        const presentCount = document.getElementById('present-count');
        const absentCount = document.getElementById('absent-count');
        const lateCount = document.getElementById('late-count');
        const percentage = document.getElementById('percentage');
        
        if (presentCount) presentCount.textContent = stats.present;
        if (absentCount) absentCount.textContent = stats.absent;
        if (lateCount) lateCount.textContent = stats.late;
        
        const percentageValue = stats.total > 0 ? ((stats.present / stats.total) * 100).toFixed(1) : 0;
        if (percentage) percentage.textContent = percentageValue + '%';
    }

    showToast(message, type = 'info') {
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;

        // Add to container
        const container = document.getElementById('toast-container') || this.createToastContainer();
        container.appendChild(toast);

        // Show toast
        const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
        bsToast.show();

        // Remove after hide
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '9998'; // Below loading overlay
        document.body.appendChild(container);
        return container;
    }

    showLoading(message = 'Loading...') {
        // Prevent duplicate loading overlays
        this.hideLoading();
        
        // Create loading overlay
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'position-fixed top-0 start-0 w-100 h-100 d-flex justify-content-center align-items-center bg-dark bg-opacity-75';
        overlay.style.zIndex = '9999';
        overlay.innerHTML = `
            <div class="bg-white p-5 rounded shadow text-center" style="min-width: 300px;">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mb-0 fw-bold">${message}</p>
                <button id="cancel-loading" class="btn btn-sm btn-outline-danger mt-3">Cancel</button>
            </div>
        `;
        
        document.body.appendChild(overlay);
        
        // Add cancel button functionality
        const cancelBtn = overlay.querySelector('#cancel-loading');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => {
                this.hideLoading();
                this.showToast('Operation cancelled', 'warning');
            });
        }
        
        // Prevent interaction with content behind
        overlay.style.pointerEvents = 'auto';
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.remove();
        }
        this.isLoading = false;
    }

    setupEventListeners() {
        // Global error handler to ensure loading is hidden
        window.addEventListener('error', () => {
            this.hideLoading();
        });
        
        window.addEventListener('unhandledrejection', () => {
            this.hideLoading();
        });
    }

    async handleFaceCapture() {
        // Prevent multiple captures
        if (this.isLoading) {
            console.log('⚠️ Face capture already in progress');
            return;
        }

        this.isLoading = true;
        this.showLoading('Processing face...');

        try {
            // Capture image from camera
            const imageData = await this.camera.captureImage();
            
            if (!imageData) {
                throw new Error('Failed to capture image');
            }
            
            console.log('✅ Image captured, sending to API...');
            
            let result;
            const regData = localStorage.getItem('registration_data');
            
            if (regData) {
                // Registration face capture
                const parsedRegData = JSON.parse(regData);
                result = await this.api.registerUser(parsedRegData, [imageData]);
                
                console.log('✅ Registration API result:', result);
                
                if (result && result.success) {
                    this.showToast('Registration successful!', 'success');
                    localStorage.removeItem('registration_data');
                    
                    // Reset UI
                    const title = document.querySelector('#login-page .card-header h4');
                    const header = document.querySelector('#login-page .card-header');
                    const captureBtn = document.querySelector('#capture-btn');
                    
                    if (title) title.textContent = '🔐 Face Login';
                    if (header) header.className = 'card-header bg-primary text-white';
                    if (captureBtn) captureBtn.innerHTML = '⚡ Capture & Login';
                    
                    // Navigate home after delay
                    setTimeout(() => {
                        this.navigateTo('home');
                    }, 2000);
                } else {
                    this.showToast(result?.error || 'Registration failed', 'error');
                }
            } else {
                // Login request
                result = await this.api.faceLogin(imageData);
                
                if (result && result.success) {
                    this.currentUser = result.user;
                    this.showLoginSuccess(result.user);
                } else {
                    this.showToast(result?.message || 'Login failed. Please try again.', 'error');
                }
            }
            
        } catch (error) {
            console.error('❌ Face capture error:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.isLoading = false;
            this.hideLoading();
        }
    }

    showLoginSuccess(user) {
        const loginDetails = document.getElementById('login-details');
        const loginResults = document.getElementById('login-results');
        
        if (loginDetails && loginResults) {
            loginDetails.innerHTML = `
                <div class="alert alert-success">
                    <h5>✅ Login Successful!</h5>
                    <p><strong>Name:</strong> ${user.full_name}</p>
                    <p><strong>User ID:</strong> ${user.user_id}</p>
                    <p><strong>Role:</strong> ${user.user_type}</p>
                    <p><strong>Confidence:</strong> ${user.confidence ? (user.confidence * 100).toFixed(1) + '%' : 'N/A'}</p>
                    <p><strong>Time:</strong> ${new Date().toLocaleTimeString()}</p>
                </div>
            `;
            
            loginResults.style.display = 'block';
            
            // Auto-navigate to attendance page for students
            if (user.user_type === 'student') {
                setTimeout(async () => {
                    this.navigateTo('attendance');
                    // Auto-fetch attendance after navigation
                    setTimeout(async () => {
                        await this.fetchAttendance();
                    }, 100);
                }, 3000);
            }
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Ensure Bootstrap is loaded
    if (typeof bootstrap === 'undefined') {
        console.error('❌ Bootstrap not loaded!');
        return;
    }
    
    // Initialize the app
    window.app = new FaceRecognitionApp();
    
    // Global error handler
    window.addEventListener('error', (event) => {
        console.error('❌ Global error:', event.error);
        window.app.hideLoading();
    });
    
    window.addEventListener('unhandledrejection', (event) => {
        console.error('❌ Unhandled promise rejection:', event.reason);
        window.app.hideLoading();
    });
});