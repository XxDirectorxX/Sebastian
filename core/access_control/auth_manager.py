"""
Authentication and authorization manager for Sebastian assistant.
"""
import logging
import yaml
import os
from typing import Dict, Any, Optional, List

from core.access_control.face_auth import FaceAuthenticator
from core.access_control.voice_auth import VoiceAuthenticator

logger = logging.getLogger(__name__)

class AuthenticationManager:
    """
    Manages user authentication and authorization across multiple methods.
    Coordinates between face recognition and voice recognition systems.
    """
    
    def __init__(self, config_path: str = "core/config/auth_config.yaml"):
        """
        Initialize authentication manager.
        
        Args:
            config_path: Path to authentication configuration file
        """
        self.face_auth = FaceAuthenticator()
        self.voice_auth = VoiceAuthenticator()
        self.authenticated_user = None
        self.auth_level = 0  # 0: None, 1: Basic, 2: Full
        self.user_permissions = {}
        
        # Load configuration
        self.config = self._load_config(config_path)
        logger.info("Authentication manager initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load authentication configuration from file."""
        default_config = {
            'require_auth': True,
            'methods': ['face', 'voice'],
            'auth_timeout': 3600,  # 1 hour
            'users': {
                'admin': {
                    'level': 2,
                    'permissions': ['all']
                },
                'user': {
                    'level': 1,
                    'permissions': ['basic', 'query']
                }
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded authentication configuration from {config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading auth config: {e}")
                return default_config
        else:
            logger.warning(f"Auth config not found at {config_path}, using defaults")
            return default_config
    
    def authenticate(self, method: Optional[str] = None, audio_data: Any = None) -> Dict[str, Any]:
        """
        Authenticate user using specified method or available methods.
        
        Args:
            method: Authentication method ('face', 'voice', or None for auto)
            audio_data: Audio data for voice authentication
            
        Returns:
            Dict with authentication results
        """
        if not self.config.get('require_auth', True):
            # Authentication disabled
            self.authenticated_user = "default_user"
            self.auth_level = 1
            return {
                'authenticated': True,
                'user': self.authenticated_user,
                'method': 'none_required',
                'level': self.auth_level
            }
        
        # Determine authentication method
        methods = self.config.get('methods', ['face', 'voice'])
        if method is None:
            method = methods[0] if methods else 'face'
        
        # Perform authentication
        if method == 'face':
            auth_result = self.face_auth.authenticate()
        elif method == 'voice' and audio_data is not None:
            auth_result = self.voice_auth.authenticate(audio_data)
        else:
            return {
                'authenticated': False,
                'error': f"Invalid authentication method: {method}"
            }
        
        # Update authentication state if successful
        if auth_result.get('authenticated', False):
            self.authenticated_user = auth_result['user']
            
            # Get user level
            user_config = self.config.get('users', {}).get(self.authenticated_user, {})
            self.auth_level = user_config.get('level', 1)
            self.user_permissions = user_config.get('permissions', ['basic'])
            
            auth_result['level'] = self.auth_level
            auth_result['permissions'] = self.user_permissions
            
            logger.info(f"User {self.authenticated_user} authenticated with level {self.auth_level}")
        else:
            logger.warning(f"Authentication failed: {auth_result.get('error', 'Unknown error')}")
        
        return auth_result
    
    def register_user(self, name: str, method: str = 'face', audio_data: Any = None, 
                      level: int = 1, permissions: List[str] = None) -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            name: User name
            method: Registration method ('face' or 'voice')
            audio_data: Audio data for voice registration
            level: Authentication level to assign
            permissions: List of permissions to grant
            
        Returns:
            Dict with registration results
        """
        if not permissions:
            permissions = ['basic', 'query']
        
        # Register user biometrics
        if method == 'face':
            result = self.face_auth.register_user(name)
        elif method == 'voice' and audio_data is not None:
            result = self.voice_auth.register_voice(audio_data, name)
        else:
            return {
                'success': False,
                'error': f"Invalid registration method: {method}"
            }
        
        # If registration successful, update configuration
        if result.get('success', False):
            # Update user configuration
            if 'users' not in self.config:
                self.config['users'] = {}
            
            self.config['users'][name] = {
                'level': level,
                'permissions': permissions
            }
            
            # Save configuration (in a real implementation)
            logger.info(f"Registered user {name} with level {level}")
        
        return result
    
    def check_permission(self, permission: str) -> bool:
        """
        Check if authenticated user has specified permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            bool: True if user has permission, False otherwise
        """
        if not self.authenticated_user:
            logger.warning("Permission check failed: No authenticated user")
            return False
        
        user_permissions = self.user_permissions
        
        # 'all' permission grants access to everything
        if 'all' in user_permissions:
            return True
        
        return permission in user_permissions

