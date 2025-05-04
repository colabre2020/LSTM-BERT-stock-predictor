import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional
import yaml
from functools import wraps
from flask import request, jsonify
import os

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class SecurityManager:
    def __init__(self):
        self.secret_key = config['dashboard']['secret_key']
        self._user_cache = {}
    
    def hash_password(self, password: str) -> bytes:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt)
    
    def verify_password(self, password: str, hashed: bytes) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
    
    def generate_token(self, username: str) -> str:
        """Generate a JWT token for a user"""
        payload = {
            'username': username,
            'exp': datetime.utcnow() + timedelta(days=1)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify a JWT token and return the payload"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def require_auth(self, f):
        """Decorator to require authentication for routes"""
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            
            if not token:
                return jsonify({'message': 'Missing token'}), 401
            
            try:
                token = token.split('Bearer ')[1]
                payload = self.verify_token(token)
                if not payload:
                    return jsonify({'message': 'Invalid token'}), 401
                
                # Add user info to request
                request.user = payload
                
            except Exception as e:
                return jsonify({'message': str(e)}), 401
            
            return f(*args, **kwargs)
        return decorated
    
    def create_user(self, username: str, password: str) -> bool:
        """Create a new user with hashed password"""
        if username in self._user_cache:
            return False
        
        hashed = self.hash_password(password)
        self._user_cache[username] = {
            'password': hashed,
            'created_at': datetime.utcnow()
        }
        return True
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return a token if successful"""
        user = self._user_cache.get(username)
        if not user:
            return None
        
        if not self.verify_password(password, user['password']):
            return None
        
        return self.generate_token(username)
    
    def invalidate_token(self, token: str) -> bool:
        """Invalidate a token (for logout)"""
        # In a production system, you would add the token to a blacklist
        # or use Redis to track invalidated tokens
        return True

# Create singleton instance
security_manager = SecurityManager()

# Utility functions
def init_default_users():
    """Initialize default users (for development only)"""
    if os.environ.get('FLASK_ENV') == 'development':
        security_manager.create_user('admin', 'admin')
        security_manager.create_user('user', 'user123')