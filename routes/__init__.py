# routes/__init__.py - Initialize the routes package

# Import route modules
from . import health_routes
from . import auth_routes
from . import user_routes
from . import mobile_routes
from . import test_routes

__all__ = ['health_routes', 'auth_routes', 'user_routes', 'mobile_routes', 'test_routes']

print("✅ Routes package initialized")