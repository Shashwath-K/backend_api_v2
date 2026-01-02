# routes/mobile_routes.py - Mobile app serving routes
from flask import Blueprint, send_from_directory, send_file, jsonify, current_app
import os

# Create blueprint for mobile routes
bp = Blueprint('mobile', __name__)

@bp.route('/mobile/<path:filename>')
def serve_mobile_file(filename):
    """Serve mobile app files from face_Recognition folder"""
    try:
        app = current_app
        mobile_app_path = app.config.get('MOBILE_APP_PATH', '')
        
        if not mobile_app_path or not os.path.exists(mobile_app_path):
            return jsonify({
                'success': False,
                'error': f'Mobile app folder not found',
                'path': mobile_app_path,
                'exists': os.path.exists(mobile_app_path) if mobile_app_path else False
            }), 404
        
        print(f"📱 Serving mobile file: {filename}")
        
        # Security check: prevent directory traversal
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            return jsonify({
                'success': False,
                'error': 'Invalid file path'
            }), 400
        
        # Only allow specific file extensions
        allowed_extensions = {'.html', '.css', '.js', '.json', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot'}
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': f'File type not allowed: {file_ext}'
            }), 403
        
        return send_from_directory(mobile_app_path, filename)
        
    except Exception as e:
        print(f"❌ Error serving mobile file {filename}: {e}")
        return jsonify({
            'success': False,
            'error': f'Mobile file {filename} not found',
            'path': app.config.get('MOBILE_APP_PATH', '') if 'app' in locals() else 'N/A',
            'exists': os.path.exists(app.config.get('MOBILE_APP_PATH', '')) if 'app' in locals() and app.config.get('MOBILE_APP_PATH') else False
        }), 404

@bp.route('/mobile')
def serve_mobile_index():
    """Serve mobile app index"""
    try:
        app = current_app
        mobile_app_path = app.config.get('MOBILE_APP_PATH', '')
        
        if not mobile_app_path:
            return jsonify({
                'success': False,
                'error': 'Mobile app path not configured',
                'current_dir': os.path.dirname(__file__) if '__file__' in locals() else 'N/A'
            }), 404
        
        print(f"📱 Serving mobile index from: {mobile_app_path}")
        
        if not os.path.exists(mobile_app_path):
            return jsonify({
                'success': False,
                'error': 'Mobile app folder not found',
                'path': mobile_app_path,
                'current_dir': os.path.dirname(__file__) if '__file__' in locals() else 'N/A'
            }), 404
        
        index_path = os.path.join(mobile_app_path, 'index.html')
        if not os.path.exists(index_path):
            # Try other common index file names
            for index_file in ['index.htm', 'default.html', 'main.html']:
                alt_path = os.path.join(mobile_app_path, index_file)
                if os.path.exists(alt_path):
                    index_path = alt_path
                    break
            
            if not os.path.exists(index_path):
                # List available files to help debugging
                try:
                    available_files = os.listdir(mobile_app_path)
                except:
                    available_files = []
                
                return jsonify({
                    'success': False,
                    'error': 'index.html not found in mobile app folder',
                    'index_path': index_path,
                    'available_files': available_files[:20],  # Show first 20 files
                    'total_files': len(available_files)
                }), 404
        
        return send_from_directory(mobile_app_path, os.path.basename(index_path))
        
    except Exception as e:
        print(f"❌ Error serving mobile index: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Mobile app not available',
            'details': str(e)
        }), 404

@bp.route('/mobile/health')
def mobile_health():
    """Health check specifically for mobile app"""
    try:
        app = current_app
        mobile_app_path = app.config.get('MOBILE_APP_PATH', '')
        
        mobile_files = []
        if mobile_app_path and os.path.exists(mobile_app_path):
            try:
                mobile_files = os.listdir(mobile_app_path)
            except:
                mobile_files = []
        
        return jsonify({
            'success': True,
            'mobile_app': {
                'path': mobile_app_path,
                'exists': os.path.exists(mobile_app_path) if mobile_app_path else False,
                'file_count': len(mobile_files),
                'has_index': 'index.html' in mobile_files or 'index.htm' in mobile_files,
                'sample_files': mobile_files[:10]  # Show first 10 files
            },
            'api_endpoints': {
                'login': '/api/login',
                'register': '/api/register',
                'users': '/api/users',
                'attendance': '/api/attendance/{student_id}'
            },
            'status': 'ready'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'mobile_app': {
                'path': 'N/A',
                'exists': False
            }
        }), 500

# Serve common mobile assets from root for backward compatibility
@bp.route('/<path:filename>')
def serve_root_file(filename):
    """Serve files that mobile app expects at root (for backward compatibility)"""
    try:
        app = current_app
        mobile_app_path = app.config.get('MOBILE_APP_PATH', '')
        
        if not mobile_app_path or not os.path.exists(mobile_app_path):
            return jsonify({'error': 'Mobile app folder not found'}), 404
        
        # Only serve specific file types for security
        allowed_extensions = {'.css', '.js', '.json', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf', '.eot'}
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'File type not allowed'}), 403
        
        # Security check
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            return jsonify({'error': 'Invalid file path'}), 400
        
        return send_from_directory(mobile_app_path, filename)
        
    except:
        return jsonify({'error': 'File not found'}), 404

@bp.route('/mobile/manifest.json')
def serve_mobile_manifest():
    """Serve mobile app manifest if exists"""
    try:
        app = current_app
        mobile_app_path = app.config.get('MOBILE_APP_PATH', '')
        
        if not mobile_app_path or not os.path.exists(mobile_app_path):
            return jsonify({'error': 'Mobile app folder not found'}), 404
        
        manifest_path = os.path.join(mobile_app_path, 'manifest.json')
        if os.path.exists(manifest_path):
            return send_from_directory(mobile_app_path, 'manifest.json')
        else:
            # Return a default manifest
            return jsonify({
                "name": "Face Recognition Attendance",
                "short_name": "FaceAttendance",
                "start_url": "/mobile",
                "display": "standalone",
                "background_color": "#ffffff",
                "theme_color": "#000000",
                "icons": []
            })
            
    except:
        return jsonify({'error': 'Manifest not available'}), 404

@bp.route('/mobile/service-worker.js')
def serve_service_worker():
    """Serve service worker if exists"""
    try:
        app = current_app
        mobile_app_path = app.config.get('MOBILE_APP_PATH', '')
        
        if not mobile_app_path or not os.path.exists(mobile_app_path):
            return jsonify({'error': 'Mobile app folder not found'}), 404
        
        sw_path = os.path.join(mobile_app_path, 'service-worker.js')
        if os.path.exists(sw_path):
            return send_from_directory(mobile_app_path, 'service-worker.js')
        else:
            # Return a simple service worker
            return """
// Simple service worker for Face Recognition Attendance
self.addEventListener('install', event => {
  console.log('Service worker installed');
});

self.addEventListener('fetch', event => {
  event.respondWith(fetch(event.request));
});
""", 200, {'Content-Type': 'application/javascript'}
            
    except:
        return jsonify({'error': 'Service worker not available'}), 404

@bp.route('/mobile/config')
def mobile_config():
    """Get mobile app configuration"""
    try:
        app = current_app
        detector = app.config.get('detector')
        encoder = app.config.get('encoder')
        db = app.config.get('db')
        
        return jsonify({
            'success': True,
            'config': {
                'api_base_url': '/api',
                'face_recognition': {
                    'detector_available': detector is not None,
                    'encoder_available': encoder is not None,
                    'database_available': db is not None
                },
                'features': {
                    'login': True,
                    'registration': True,
                    'attendance_tracking': True,
                    'user_management': True
                },
                'version': '1.0.0',
                'environment': 'production' if not app.debug else 'development'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'config': None
        }), 500