# server_main.py - Main entry point for Face Recognition Attendance API
from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import route modules
from routes import health_routes, auth_routes, user_routes, mobile_routes, test_routes

# Define mobile app path (go up one level, then into face_Recognition)
MOBILE_APP_PATH = os.path.join(os.path.dirname(__file__), '..', 'face_Recognition')
print(f"📱 Mobile app path: {MOBILE_APP_PATH}")
print(f"📱 Path exists: {os.path.exists(MOBILE_APP_PATH)}")

print("🔧 Initializing face recognition system...")

# Global variables for components that will be shared across routes
detector = None
encoder = None
db = None

def create_app():
    """Create and configure the Flask application"""
    try:
        app = Flask(__name__)
        CORS(app)
        print("✅ Flask app initialized")
    except Exception as e:
        print(f"❌ Error initializing Flask: {e}")
        # Try alternative initialization
        app = Flask("FaceRecognitionAPI")
        CORS(app)
    
    # Import components after app creation
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
    
    # Initialize global components
    global detector, encoder, db
    
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
    
    # Register blueprints/routes
    register_routes(app)
    
    # Set global variables for routes
    app.config['MOBILE_APP_PATH'] = MOBILE_APP_PATH
    app.config['detector'] = detector
    app.config['encoder'] = encoder
    app.config['db'] = db
    
    return app

def register_routes(app):
    """Register all route blueprints with the app"""
    # Register health routes
    app.register_blueprint(health_routes.bp)
    
    # Register authentication routes
    app.register_blueprint(auth_routes.bp)
    
    # Register user routes
    app.register_blueprint(user_routes.bp)
    
    # Register mobile routes
    app.register_blueprint(mobile_routes.bp)
    
    # Register test routes
    app.register_blueprint(test_routes.bp)
    
    # Root endpoint
    @app.route('/')
    def index():
        """API information page"""
        mobile_files = []
        mobile_app_path = app.config.get('MOBILE_APP_PATH', MOBILE_APP_PATH)
        if os.path.exists(mobile_app_path):
            try:
                mobile_files = os.listdir(mobile_app_path)
            except:
                mobile_files = []
        
        return jsonify({
            'name': 'Face Recognition Attendance API',
            'version': '1.0.0',
            'description': 'API for face recognition-based attendance system',
            'status': 'running',
            'mobile_app': {
                'path': mobile_app_path,
                'exists': os.path.exists(mobile_app_path),
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
    
    # Create Flask app
    app = create_app()
    
    print("\n🌐 Starting Flask API Server...")
    print("🔧 API Base URL: http://localhost:5000/")
    print("🩺 Health Check: http://localhost:5000/api/health")
    print("🧪 Test Endpoint: http://localhost:5000/api/test")
    print("📋 Test Interface: http://localhost:5000/test.html")
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
    print("   GET  /api/attendance/<student_id> - Get attendance records")
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