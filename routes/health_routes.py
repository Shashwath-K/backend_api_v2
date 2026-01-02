# routes/health_routes.py - Health check and system status routes
from flask import Blueprint, jsonify, current_app
from datetime import datetime

# Create blueprint for health routes
bp = Blueprint('health', __name__, url_prefix='/api')

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    app = current_app
    
    # Get components from app config
    detector = app.config.get('detector')
    encoder = app.config.get('encoder')
    db = app.config.get('db')
    mobile_app_path = app.config.get('MOBILE_APP_PATH', '')
    
    # Check database connection if available
    db_status = False
    if db is not None:
        if hasattr(db, 'is_connected') and callable(db.is_connected):
            db_status = db.is_connected()
        else:
            db_status = True  # Assume connected if no is_connected method
    
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'detector': detector is not None,
            'encoder': encoder is not None,
            'database': db_status
        },
        'system_info': {
            'version': '1.0.0',
            'api_base': '/api',
            'mobile_app_path': mobile_app_path,
            'mobile_app_exists': len(mobile_app_path) > 0
        },
        'endpoints': [
            {'method': 'GET', 'path': '/api/health', 'description': 'System health check'},
            {'method': 'POST', 'path': '/api/login', 'description': 'Face login authentication'},
            {'method': 'POST', 'path': '/api/register', 'description': 'User registration with face'},
            {'method': 'GET', 'path': '/api/users', 'description': 'List all registered users'},
            {'method': 'GET', 'path': '/api/attendance/<student_id>', 'description': 'Get attendance records'},
            {'method': 'GET', 'path': '/api/test', 'description': 'Test endpoint'}
        ]
    })

@bp.route('/status', methods=['GET'])
def system_status():
    """Detailed system status endpoint"""
    app = current_app
    
    # Get components from app config
    detector = app.config.get('detector')
    encoder = app.config.get('encoder')
    db = app.config.get('db')
    
    # Gather detailed status
    detector_status = {
        'available': detector is not None,
        'type': detector.__class__.__name__ if detector else 'None',
        'method': getattr(detector, 'method', 'unknown') if detector else 'N/A'
    }
    
    encoder_status = {
        'available': encoder is not None,
        'type': encoder.__class__.__name__ if encoder else 'None',
        'use_deepface': getattr(encoder, 'use_deepface', False) if encoder else False
    }
    
    db_status = {
        'available': db is not None,
        'connected': False,
        'type': db.__class__.__name__ if db else 'None'
    }
    
    # Check database connection
    if db is not None:
        if hasattr(db, 'is_connected') and callable(db.is_connected):
            db_status['connected'] = db.is_connected()
        else:
            db_status['connected'] = True  # Assume connected if no is_connected method
    
    return jsonify({
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'system': {
            'name': 'Face Recognition Attendance System',
            'version': '1.0.0',
            'status': 'operational',
            'uptime': 'N/A'  # Would need to track startup time
        },
        'components': {
            'face_detector': detector_status,
            'face_encoder': encoder_status,
            'database': db_status
        },
        'recommendations': []
    })