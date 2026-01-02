# routes/test_routes.py - Test routes and HTML serving
from flask import Blueprint, jsonify, send_file, current_app
from datetime import datetime
import os

# Create blueprint for test routes
bp = Blueprint('test', __name__, url_prefix='/api')

@bp.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify server is running"""
    app = current_app
    detector = app.config.get('detector')
    encoder = app.config.get('encoder')
    db = app.config.get('db')
    
    return jsonify({
        'success': True,
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'detector': detector is not None,
            'encoder': encoder is not None,
            'database': db is not None
        },
        'endpoints': [
            {'method': 'GET', 'path': '/api/health', 'description': 'System health check'},
            {'method': 'POST', 'path': '/api/login', 'description': 'Face login authentication'},
            {'method': 'POST', 'path': '/api/register', 'description': 'User registration with face'},
            {'method': 'GET', 'path': '/api/users', 'description': 'List all registered users'},
            {'method': 'GET', 'path': '/api/attendance/<student_id>', 'description': 'Get attendance records'},
            {'method': 'GET', 'path': '/api/test', 'description': 'Test endpoint'}
        ],
        'quick_test': {
            'health_check': '/api/health',
            'list_users': '/api/users',
            'mobile_app': '/mobile'
        }
    })

@bp.route('/test/detector', methods=['GET'])
def test_detector():
    """Test face detector functionality"""
    try:
        app = current_app
        detector = app.config.get('detector')
        
        if detector is None:
            return jsonify({
                'success': False,
                'error': 'Face detector not available',
                'detector_type': 'None'
            }), 503
        
        detector_info = {
            'type': detector.__class__.__name__,
            'method': getattr(detector, 'method', 'unknown'),
            'available_methods': getattr(detector, 'available_methods', []),
            'model_loaded': getattr(detector, 'model_loaded', False)
        }
        
        return jsonify({
            'success': True,
            'message': 'Face detector is available',
            'detector': detector_info,
            'test_image': 'Send a POST request to /api/login with image data to test detection'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'detector': None
        }), 500

@bp.route('/test/encoder', methods=['GET'])
def test_encoder():
    """Test face encoder functionality"""
    try:
        app = current_app
        encoder = app.config.get('encoder')
        
        if encoder is None:
            return jsonify({
                'success': False,
                'error': 'Face encoder not available',
                'encoder_type': 'None'
            }), 503
        
        encoder_info = {
            'type': encoder.__class__.__name__,
            'use_deepface': getattr(encoder, 'use_deepface', False),
            'embedding_dim': getattr(encoder, 'embedding_dim', 256),
            'model_name': getattr(encoder, 'model_name', 'unknown')
        }
        
        return jsonify({
            'success': True,
            'message': 'Face encoder is available',
            'encoder': encoder_info,
            'test_note': 'Encoder works with detector during login/registration'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'encoder': None
        }), 500

@bp.route('/test/database', methods=['GET'])
def test_database():
    """Test database connectivity"""
    try:
        app = current_app
        db = app.config.get('db')
        
        if db is None:
            return jsonify({
                'success': False,
                'error': 'Database not available',
                'database_type': 'None'
            }), 503
        
        db_info = {
            'type': db.__class__.__name__,
            'connected': False,
            'methods_available': []
        }
        
        # Check connection
        if hasattr(db, 'is_connected') and callable(db.is_connected):
            db_info['connected'] = db.is_connected()
        
        # List available methods
        for method_name in dir(db):
            if not method_name.startswith('_') and callable(getattr(db, method_name)):
                db_info['methods_available'].append(method_name)
        
        # Test basic operations
        test_results = {}
        if hasattr(db, 'get_all_users'):
            try:
                users = db.get_all_users()
                test_results['get_all_users'] = {
                    'success': True,
                    'count': len(users) if users else 0
                }
            except Exception as e:
                test_results['get_all_users'] = {
                    'success': False,
                    'error': str(e)
                }
        
        return jsonify({
            'success': True,
            'message': 'Database connection test',
            'database': db_info,
            'test_results': test_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'database': None
        }), 500

@bp.route('/test/system', methods=['GET'])
def test_system():
    """Comprehensive system test"""
    try:
        app = current_app
        detector = app.config.get('detector')
        encoder = app.config.get('encoder')
        db = app.config.get('db')
        
        tests = {}
        
        # Test 1: Detector
        tests['detector'] = {
            'available': detector is not None,
            'details': {
                'type': detector.__class__.__name__ if detector else 'None',
                'method': getattr(detector, 'method', 'N/A') if detector else 'N/A'
            } if detector else None
        }
        
        # Test 2: Encoder
        tests['encoder'] = {
            'available': encoder is not None,
            'details': {
                'type': encoder.__class__.__name__ if encoder else 'None',
                'embedding_dim': getattr(encoder, 'embedding_dim', 256) if encoder else 'N/A'
            } if encoder else None
        }
        
        # Test 3: Database
        tests['database'] = {
            'available': db is not None,
            'connected': False,
            'details': {
                'type': db.__class__.__name__ if db else 'None'
            } if db else None
        }
        if db and hasattr(db, 'is_connected'):
            tests['database']['connected'] = db.is_connected()
        
        # Test 4: File system
        mobile_app_path = app.config.get('MOBILE_APP_PATH', '')
        tests['file_system'] = {
            'mobile_app_path': mobile_app_path,
            'exists': os.path.exists(mobile_app_path) if mobile_app_path else False,
            'has_index': os.path.exists(os.path.join(mobile_app_path, 'index.html')) if mobile_app_path and os.path.exists(mobile_app_path) else False
        }
        
        # Calculate overall status
        critical_tests = ['detector', 'encoder']
        critical_passed = all(tests[test]['available'] for test in critical_tests if test in tests)
        
        overall_status = 'operational' if critical_passed else 'degraded'
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'system_status': overall_status,
            'tests': tests,
            'recommendations': [
                'Ensure all critical components (detector, encoder) are available for full functionality',
                'Check database connection if user management is required',
                'Verify mobile app files are in the correct location for web interface'
            ] if overall_status == 'degraded' else ['All systems operational']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'system_status': 'failed',
            'tests': {}
        }), 500

# HTML serving routes (outside API prefix)
@bp.route('/test.html')
def serve_test_html():
    """Serve the test HTML page"""
    try:
        # Look for test.html in current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_html_path = os.path.join(current_dir, '..', 'test.html')
        
        if os.path.exists(test_html_path):
            return send_file(test_html_path)
        else:
            # Create a simple test page if none exists
            simple_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition API Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { display: inline-block; padding: 5px 10px; background: #4CAF50; color: white; border-radius: 3px; }
        .method.get { background: #2196F3; }
        .method.post { background: #4CAF50; }
        button { padding: 10px 20px; margin: 5px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #1976D2; }
        .result { margin-top: 20px; padding: 15px; background: #f9f9f9; border-left: 4px solid #2196F3; }
    </style>
</head>
<body>
    <h1>Face Recognition API Test Interface</h1>
    <p>Use this page to test the API endpoints.</p>
    
    <div class="endpoint">
        <span class="method get">GET</span> <strong>/api/health</strong>
        <button onclick="testEndpoint('/api/health')">Test</button>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span> <strong>/api/test</strong>
        <button onclick="testEndpoint('/api/test')">Test</button>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span> <strong>/api/users</strong>
        <button onclick="testEndpoint('/api/users')">Test</button>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span> <strong>/api/test/system</strong>
        <button onclick="testEndpoint('/api/test/system')">Test System</button>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span> <strong>/mobile</strong>
        <button onclick="window.open('/mobile', '_blank')">Open Mobile App</button>
    </div>
    
    <div id="result" class="result"></div>
    
    <script>
        async function testEndpoint(endpoint) {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = 'Testing...';
            
            try {
                const response = await fetch(endpoint);
                const data = await response.json();
                resultDiv.innerHTML = '<h3>Response:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
            } catch (error) {
                resultDiv.innerHTML = '<h3>Error:</h3><pre>' + error + '</pre>';
            }
        }
    </script>
</body>
</html>
"""
            return simple_html, 200, {'Content-Type': 'text/html'}
            
    except Exception as e:
        print(f"❌ Error serving test.html: {e}")
        return jsonify({
            'success': False,
            'error': 'test.html not found',
            'message': 'Make sure test.html is in the same directory as server_main.py'
        }), 404

@bp.route('/test/echo', methods=['POST'])
def echo_test():
    """Echo back request data for testing"""
    try:
        data = current_app.request.json
        
        if not data:
            return jsonify({
                'success': True,
                'message': 'No data received',
                'received': None,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({
            'success': True,
            'message': 'Data received successfully',
            'received': data,
            'timestamp': datetime.now().isoformat(),
            'headers': dict(current_app.request.headers)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'received': None
        }), 400

@bp.route('/test/performance', methods=['GET'])
def performance_test():
    """Performance testing endpoint"""
    import time
    
    start_time = time.time()
    
    app = current_app
    detector = app.config.get('detector')
    encoder = app.config.get('encoder')
    db = app.config.get('db')
    
    # Simulate some operations
    operations = []
    
    # Operation 1: Component check
    op1_start = time.time()
    components = {
        'detector': detector is not None,
        'encoder': encoder is not None,
        'database': db is not None
    }
    op1_time = time.time() - op1_start
    operations.append({'name': 'Component check', 'time_ms': round(op1_time * 1000, 2)})
    
    # Operation 2: Database ping (if available)
    if db and hasattr(db, 'is_connected'):
        op2_start = time.time()
        db_connected = db.is_connected()
        op2_time = time.time() - op2_start
        operations.append({'name': 'Database ping', 'time_ms': round(op2_time * 1000, 2), 'connected': db_connected})
    
    total_time = time.time() - start_time
    
    return jsonify({
        'success': True,
        'performance': {
            'total_time_ms': round(total_time * 1000, 2),
            'operations': operations,
            'requests_per_second': round(1 / total_time, 2) if total_time > 0 else 0,
            'memory_usage_mb': 'N/A'  # Could add psutil for actual memory usage
        },
        'recommendations': [
            'All response times under 100ms are considered good',
            'Database operations should be under 50ms for optimal performance',
            'Consider caching frequent requests'
        ]
    })