CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ==================== FACE_TEMPLATES TABLE ====================
CREATE TABLE face_templates (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(100) NOT NULL UNIQUE,
    person_name VARCHAR(255),
    embedding VECTOR(256) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_face_templates_person_id ON face_templates(person_id);
CREATE INDEX idx_created_at ON face_templates(created_at);

-- Create IVFFLAT index for vector similarity search
CREATE INDEX idx_face_templates_embedding ON face_templates 
USING ivfflat (embedding vector_cosine_ops);

-- ==================== Users TABLE ====================
CREATE TABLE users (
    user_id VARCHAR(100) PRIMARY KEY,
    person_id VARCHAR(100) NOT NULL UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    user_type VARCHAR(20) NOT NULL CHECK (user_type IN ('student', 'faculty', 'staff', 'admin')),
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    date_of_birth DATE,
    registration_date DATE DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE,
    department VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_user_face_template 
        FOREIGN KEY (person_id) 
        REFERENCES face_templates(person_id)
        ON DELETE CASCADE
);

-- Create indexes for users table
CREATE INDEX idx_users_user_type ON users(user_type);
CREATE INDEX idx_users_department ON users(department);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_active ON users(is_active);

-- ==================== STUDENTS TABLE ====================
CREATE TABLE students (
    student_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    enrollment_number VARCHAR(50) UNIQUE,
    semester INT,
    program VARCHAR(100),
    batch_year VARCHAR(5),
    email VARCHAR(255),
    phone VARCHAR(20),
    date_of_birth DATE,
    enrollment_date DATE DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_student_user 
        FOREIGN KEY (user_id) 
        REFERENCES users(user_id)
        ON DELETE CASCADE 
);

-- Create indexes for students table
CREATE INDEX idx_students_is_active ON students(is_active);
CREATE INDEX idx_students_full_name ON students(full_name);
CREATE INDEX idx_students_email ON students(email);
CREATE INDEX idx_students_semester ON students(semester);
CREATE INDEX idx_students_program ON students(program);

-- ===================== FACULTY TABLE ====================
CREATE TABLE faculty (
    faculty_id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL UNIQUE,
    designation VARCHAR(100),
    qualification VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_faculty_user 
        FOREIGN KEY (user_id) 
        REFERENCES users(user_id)
        ON DELETE CASCADE
);

-- ==================== COURSES TABLE ====================
CREATE TABLE courses (
    course_id SERIAL PRIMARY KEY,
    course_code VARCHAR(50) UNIQUE NOT NULL,
    course_name VARCHAR(255) NOT NULL,
    department VARCHAR(100),
    credits INTEGER,
    semester INTEGER,
    academic_year VARCHAR(10),
    faculty_id VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_course_faculty 
        FOREIGN KEY (faculty_id) 
        REFERENCES faculty(faculty_id)
        ON DELETE SET NULL
);

-- ==================== ATTENDANCE SESSIONS TABLE ====================
CREATE TABLE attendance_sessions (
    session_id SERIAL PRIMARY KEY,
    session_name VARCHAR(255) NOT NULL,
    session_type VARCHAR(50) NOT NULL CHECK (session_type IN ('lecture', 'lab', 'meeting', 'event', 'shift', 'general')),
    
    location_id VARCHAR(100),
    room_number VARCHAR(50),
    
    scheduled_start TIMESTAMP NOT NULL,
    scheduled_end TIMESTAMP,
    actual_start TIMESTAMP,
    actual_end TIMESTAMP,
    
    course_id INTEGER,
    subject_name VARCHAR(255),
    faculty_in_charge VARCHAR(100),
    
    session_status VARCHAR(20) DEFAULT 'scheduled' 
        CHECK (session_status IN ('scheduled', 'ongoing', 'completed', 'cancelled')),
    expected_duration_minutes INTEGER,
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_session_course 
        FOREIGN KEY (course_id) 
        REFERENCES courses(course_id)
        ON DELETE SET NULL,
    
    CONSTRAINT fk_session_faculty 
        FOREIGN KEY (faculty_in_charge) 
        REFERENCES faculty(faculty_id)
        ON DELETE SET NULL
);

-- ==================== LEAVE REQUESTS TABLE ====================
CREATE TABLE leave_requests (
    leave_id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    person_id VARCHAR(100) NOT NULL,
    leave_type VARCHAR(50) NOT NULL CHECK (leave_type IN ('sick', 'casual', 'vacation', 'personal', 'medical', 'other')),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_days INTEGER,
    reason TEXT,
    
    status VARCHAR(20) DEFAULT 'pending' 
        CHECK (status IN ('pending', 'approved', 'rejected', 'cancelled')),
    approved_by VARCHAR(100),
    approved_at TIMESTAMP,
    rejection_reason TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_leave_user 
        FOREIGN KEY (user_id) 
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    
    CONSTRAINT fk_leave_person 
        FOREIGN KEY (person_id) 
        REFERENCES face_templates(person_id)
        ON DELETE CASCADE
);

-- ==================== STUDENT ATTENDANCE TABLE ====================
CREATE TABLE student_attendance (
    attendance_id SERIAL PRIMARY KEY,
    
    -- Student Identification
    student_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    person_id VARCHAR(100) NOT NULL,
    
    -- Academic Context
    course_id INTEGER,
    class_session_id INTEGER,
    subject_code VARCHAR(50),
    subject_name VARCHAR(255),
    
    -- Attendance Details
    attendance_date DATE NOT NULL DEFAULT CURRENT_DATE,
    attendance_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    attendance_type VARCHAR(20) NOT NULL DEFAULT 'class' 
        CHECK (attendance_type IN ('class')),
    
    -- Status and Marks
    attendance_status VARCHAR(20) NOT NULL DEFAULT 'present'
        CHECK (attendance_status IN ('present', 'absent', 'late', 'excused', 'half_day', 'leave')),
    attendance_mark DECIMAL(3,1) DEFAULT 1.0,
    
    -- Time Tracking
    scheduled_start_time TIME,
    scheduled_end_time TIME,
    actual_checkin_time TIME,
    actual_checkout_time TIME,
    late_minutes INTEGER DEFAULT 0,
    early_departure_minutes INTEGER DEFAULT 0,
    
    -- Face Recognition Details
    face_template_id INTEGER,
    confidence_score DECIMAL(5,4),
    authentication_method VARCHAR(20) DEFAULT 'face' 
        CHECK (authentication_method IN ('face')),
    
    -- Session Information
    semester INTEGER,
    academic_year VARCHAR(10),
    period_number INTEGER,
    room_number VARCHAR(50),
    
    -- Teacher/Faculty Tracking
    faculty_id VARCHAR(100),
    marked_by VARCHAR(100),
    
    -- Leave Integration
    leave_id INTEGER,
    
    -- Correction/Override Information
    is_corrected BOOLEAN DEFAULT FALSE,
    corrected_by VARCHAR(100),
    correction_reason TEXT,
    original_status VARCHAR(20),
    
    -- Verification
    is_verified BOOLEAN DEFAULT FALSE,
    verified_by VARCHAR(100),
    verified_at TIMESTAMP,
    
    -- Metadata
    notes TEXT,
    device_location VARCHAR(255),
    
    -- Audit Trail
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign Key Constraints
    CONSTRAINT fk_student_attendance_student 
        FOREIGN KEY (student_id) 
        REFERENCES students(student_id)
        ON DELETE CASCADE,
    
    CONSTRAINT fk_student_attendance_user 
        FOREIGN KEY (user_id) 
        REFERENCES users(user_id)
        ON DELETE CASCADE,
    
    CONSTRAINT fk_student_attendance_person 
        FOREIGN KEY (person_id) 
        REFERENCES face_templates(person_id)
        ON DELETE CASCADE,
    
    CONSTRAINT fk_student_attendance_course 
        FOREIGN KEY (course_id) 
        REFERENCES courses(course_id)
        ON DELETE SET NULL,
    
    CONSTRAINT fk_student_attendance_session 
        FOREIGN KEY (class_session_id) 
        REFERENCES attendance_sessions(session_id)
        ON DELETE SET NULL,
    
    CONSTRAINT fk_student_attendance_faculty 
        FOREIGN KEY (faculty_id) 
        REFERENCES faculty(faculty_id)
        ON DELETE SET NULL,
    
    CONSTRAINT fk_student_attendance_face_template 
        FOREIGN KEY (face_template_id) 
        REFERENCES face_templates(id)
        ON DELETE SET NULL,
    
    CONSTRAINT fk_student_attendance_leave 
        FOREIGN KEY (leave_id) 
        REFERENCES leave_requests(leave_id)
        ON DELETE SET NULL
);

-- ==================== INDEXES FOR STUDENT ATTENDANCE ====================
CREATE INDEX idx_student_attendance_student_date ON student_attendance(student_id, attendance_date);
CREATE INDEX idx_student_attendance_course_date ON student_attendance(course_id, attendance_date);
CREATE INDEX idx_student_attendance_date_status ON student_attendance(attendance_date, attendance_status);
CREATE INDEX idx_student_attendance_semester ON student_attendance(semester, academic_year);
CREATE INDEX idx_student_attendance_faculty_date ON student_attendance(faculty_id, attendance_date);
CREATE INDEX idx_student_attendance_person ON student_attendance(person_id);
CREATE INDEX idx_student_attendance_time ON student_attendance(attendance_time);
CREATE INDEX idx_student_attendance_leave ON student_attendance(leave_id) WHERE leave_id IS NOT NULL;
CREATE INDEX idx_student_attendance_corrected ON student_attendance(is_corrected);
CREATE INDEX idx_student_attendance_verified ON student_attendance(is_verified);
CREATE INDEX idx_student_attendance_type_date ON student_attendance(attendance_type, attendance_date);
CREATE INDEX idx_student_attendance_academic ON student_attendance(student_id, semester, academic_year, course_id);

-- ==================== ADDITIONAL INDEXES ====================
CREATE INDEX idx_courses_code ON courses(course_code);
CREATE INDEX idx_courses_semester ON courses(semester, academic_year);
CREATE INDEX idx_sessions_type ON attendance_sessions(session_type);
CREATE INDEX idx_sessions_date ON attendance_sessions(scheduled_start);
CREATE INDEX idx_sessions_faculty ON attendance_sessions(faculty_in_charge);
CREATE INDEX idx_leave_requests_user ON leave_requests(user_id);
CREATE INDEX idx_leave_requests_dates ON leave_requests(start_date, end_date);
CREATE INDEX idx_leave_requests_status ON leave_requests(status);