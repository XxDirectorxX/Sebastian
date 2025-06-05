"""
Face authentication for Sebastian assistant.
"""
import logging
import time
from typing import Dict, Any, Optional, List, Tuple

from core.vision.vision_manager import VisionManager

logger = logging.getLogger(__name__)

class FaceAuthenticator:
    """
    Provides face-based authentication using the vision system.
    """
    
    def __init__(self):
        """Initialize face authenticator with vision manager."""
        self.vision_manager = VisionManager()
        self.auth_threshold = 0.7  # Minimum confidence for authentication
        self.recent_auths = {}  # Recent authentication results (to avoid constant scanning)
        self.auth_cache_time = 60  # Seconds to cache authentication results
        logger.info("Face authenticator initialized")
    
    def authenticate(self, required_user: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate user via facial recognition.
        
        Args:
            required_user: If provided, only authenticate this specific user
            
        Returns:
            Dict with authentication results
        """
        # Check cache first
        if required_user in self.recent_auths:
            last_time, authenticated = self.recent_auths[required_user]
            if time.time() - last_time < self.auth_cache_time:
                logger.info(f"Using cached authentication for {required_user}")
                return {
                    'authenticated': authenticated,
                    'user': required_user if authenticated else None,
                    'method': 'face_cached',
                    'confidence': None
                }
        
        # Perform facial recognition
        success, faces = self.vision_manager.detect_faces()
        
        if not success or not faces:
            logger.warning("Face authentication failed: No face detected")
            return {
                'authenticated': False,
                'user': None,
                'method': 'face',
                'error': 'No face detected'
            }
        
        # Find the face with highest confidence
        best_match = max(faces, key=lambda f: f['confidence'])
        
        # Check if confidence meets threshold
        if best_match['confidence'] < self.auth_threshold:
            logger.warning(f"Face authentication failed: Confidence too low ({best_match['confidence']:.2f})")
            return {
                'authenticated': False,
                'user': None,
                'method': 'face',
                'confidence': best_match['confidence'],
                'error': 'Confidence too low'
            }
        
        # If specific user is required, check if matched
        if required_user and best_match['name'] != required_user:
            logger.warning(f"Face authentication failed: User mismatch (expected {required_user}, got {best_match['name']})")
            self.recent_auths[required_user] = (time.time(), False)
            return {
                'authenticated': False,
                'user': best_match['name'],
                'method': 'face',
                'confidence': best_match['confidence'],
                'error': f'User mismatch (expected {required_user})'
            }
        
        # Authentication successful
        user = best_match['name']
        logger.info(f"Face authentication successful for {user} with confidence {best_match['confidence']:.2f}")
        
        # Cache result
        if required_user:
            self.recent_auths[required_user] = (time.time(), True)
        
        return {
            'authenticated': True,
            'user': user,
            'method': 'face',
            'confidence': best_match['confidence']
        }
    
    def register_user(self, name: str) -> Dict[str, Any]:
        """
        Register a new user face.
        
        Args:
            name: Name to associate with the face
            
        Returns:
            Dict with registration results
        """
        success = self.vision_manager.register_face(name)
        
        if success:
            logger.info(f"Registered face for user {name}")
            return {
                'success': True,
                'user': name,
                'message': f"Successfully registered face for {name}"
            }
        else:
            logger.warning(f"Failed to register face for user {name}")
            return {
                'success': False,
                'user': name,
                'error': "Failed to register face"
            }