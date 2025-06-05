"""
Security manager for Sebastian assistant.

Provides comprehensive security services including:
- Authentication and authorization
- Access control enforcement
- Encryption of sensitive data
- Security audit logging
- Input validation
"""
import logging
import os
import time
import hashlib
import secrets
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Comprehensive security management for the Sebastian assistant.
    
    Handles user authentication, access control, secure storage,
    encryption, and security audit logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize security manager with configuration.
        
        Args:
            config: Security configuration settings
        """
        self.config = config
        self.access_control_enabled = config.get("access_control", True)
        self.encryption_enabled = config.get("encryption", True)
        self.security_log_path = Path(config.get("security_log", "logs/security.log"))
        self.security_log_path.parent.mkdir(exist_ok=True)
        
        # User authentication store
        self.auth_db_path = Path(config.get("auth_db", "core/security/auth_data.json"))
        self.auth_db_path.parent.mkdir(exist_ok=True)
        self.users = self._load_users()
        
        # Session management
        self.sessions = {}
        self.session_timeout = config.get("session_timeout", 3600)  # 1 hour default
        
        # Initialize encryption
        self._setup_encryption()
        
        # Initialize access control rules
        self.access_rules = self._load_access_rules()
        
        logger.info("Security manager initialized")
        self.log_security_event("system", "SecurityManager initialized")
        
    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load user authentication data."""
        if not self.auth_db_path.exists():
            # Create default admin user if no user database exists
            default_users = {
                "administrator": {
                    "password_hash": self._hash_password("administrator", "change_this_password_immediately"),
                    "salt": secrets.token_hex(16),
                    "role": "admin",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "face_id": None,
                    "voice_id": None,
                    "status": "active"
                },
                "guest": {
                    "password_hash": self._hash_password("guest", "guest"),
                    "salt": secrets.token_hex(16),
                    "role": "guest",
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "face_id": None,
                    "voice_id": None,
                    "status": "active"
                }
            }
            
            # Save default users
            with open(self.auth_db_path, 'w') as f:
                json.dump(default_users, f, indent=2)
                
            self.log_security_event("system", "Created default user accounts")
            return default_users
        
        try:
            with open(self.auth_db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load user database: {e}")
            self.log_security_event("system", f"Failed to load user database: {e}")
            return {}
            
    def _save_users(self) -> None:
        """Save user authentication data."""
        try:
            with open(self.auth_db_path, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user database: {e}")
            self.log_security_event("system", f"Failed to save user database: {e}")
            
    def _hash_password(self, username: str, password: str) -> str:
        """
        Create secure hash of password with salt.
        
        Args:
            username: Username for the account
            password: Password to hash
            
        Returns:
            Secure password hash
        """
        # Get or create salt for user
        if username in self.users and "salt" in self.users[username]:
            salt = self.users[username]["salt"]
        else:
            salt = secrets.token_hex(16)
        
        # Create hash with username, password and salt
        password_bytes = (username + password + salt).encode('utf-8')
        hash_obj = hashlib.sha3_512(password_bytes)
        return hash_obj.hexdigest()
            
    def _setup_encryption(self) -> None:
        """Initialize encryption system."""
        if not self.encryption_enabled:
            return
            
        key_path = Path(self.config.get("encryption_key", "core/security/encryption.key"))
        
        if not key_path.exists():
            # Generate a new encryption key
            key_path.parent.mkdir(exist_ok=True)
            key = Fernet.generate_key()
            
            with open(key_path, 'wb') as f:
                f.write(key)
                
            self.log_security_event("system", "Generated new encryption key")
        else:
            # Load existing key
            with open(key_path, 'rb') as f:
                key = f.read()
        
        self.cipher = Fernet(key)
        
    def _load_access_rules(self) -> Dict[str, Dict[str, List[str]]]:
        """Load access control rules."""
        rules_path = Path(self.config.get("access_rules", "core/security/access_rules.json"))
        
        if not rules_path.exists():
            # Create default rules
            default_rules = {
                "admin": {
                    "allowed_functions": ["*"],  # Admin can do everything
                    "allowed_plugins": ["*"]
                },
                "family": {
                    "allowed_functions": [
                        "query_knowledge", "set_reminder", "control_devices", 
                        "play_media", "send_message"
                    ],
                    "allowed_plugins": [
                        "alarm", "device_control", "reminder", "weather", 
                        "music", "calendar", "messaging"
                    ]
                },
                "guest": {
                    "allowed_functions": ["query_knowledge", "control_basic_devices"],
                    "allowed_plugins": ["weather", "music", "basic_control"]
                }
            }
            
            # Save default rules
            rules_path.parent.mkdir(exist_ok=True)
            with open(rules_path, 'w') as f:
                json.dump(default_rules, f, indent=2)
                
            return default_rules
        
        try:
            with open(rules_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load access rules: {e}")
            self.log_security_event("system", f"Failed to load access rules: {e}")
            return {}
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            
        Returns:
            Tuple of (success, session_token or None)
        """
        if not self.access_control_enabled:
            # If access control is disabled, auto-succeed with admin access
            session_token = self._create_session("admin", "admin")
            return True, session_token
            
        if username not in self.users:
            self.log_security_event("authentication", f"Failed login attempt for non-existent user: {username}")
            return False, None
            
        user = self.users[username]
        
        # Check if account is locked or inactive
        if user.get("status") != "active":
            self.log_security_event("authentication", f"Login attempt for inactive account: {username}")
            return False, None
            
        # Verify password
        password_hash = self._hash_password(username, password)
        if password_hash != user["password_hash"]:
            # Increment failed attempts
            if "failed_attempts" not in user:
                user["failed_attempts"] = 0
            user["failed_attempts"] += 1
            
            # Lock account after too many attempts
            if user["failed_attempts"] >= 5:
                user["status"] = "locked"
                self.log_security_event("authentication", f"Account locked due to too many failed attempts: {username}")
                
            self._save_users()
            self.log_security_event("authentication", f"Failed login attempt for user: {username}")
            return False, None
            
        # Authentication successful
        user["last_login"] = datetime.now().isoformat()
        user["failed_attempts"] = 0
        self._save_users()
        
        # Create session
        session_token = self._create_session(username, user["role"])
        self.log_security_event("authentication", f"Successful login for user: {username}")
        
        return True, session_token
        
    def authenticate_biometric(self, biometric_type: str, biometric_data: Any) -> Tuple[bool, Optional[str]]:
        """
        Authenticate user with biometric data.
        
        Args:
            biometric_type: Type of biometric ("face" or "voice")
            biometric_data: Biometric data to match
            
        Returns:
            Tuple of (success, session_token or None)
        """
        if not self.access_control_enabled:
            # If access control is disabled, auto-succeed with admin access
            session_token = self._create_session("admin", "admin")
            return True, session_token
            
        # This would be implemented with actual biometric matching
        # For now, just log the attempt
        self.log_security_event("authentication", f"Biometric authentication attempt: {biometric_type}")
        return False, None
        
    def _create_session(self, username: str, role: str) -> str:
        """
        Create a new authenticated session.
        
        Args:
            username: Username for the session
            role: User role
            
        Returns:
            Session token
        """
        # Generate secure random token
        token = secrets.token_hex(32)
        expires = datetime.now() + timedelta(seconds=self.session_timeout)
        
        # Store session
        self.sessions[token] = {
            "username": username,
            "role": role,
            "created": datetime.now(),
            "expires": expires,
            "last_activity": datetime.now()
        }
        
        return token
        
    def validate_session(self, session_token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate session token and return session info.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            Tuple of (valid, session_info or None)
        """
        if not self.access_control_enabled:
            # If access control is disabled, treat all sessions as valid admin sessions
            return True, {
                "username": "admin",
                "role": "admin",
                "created": datetime.now(),
                "expires": datetime.now() + timedelta(seconds=self.session_timeout)
            }
            
        if session_token not in self.sessions:
            return False, None
            
        session = self.sessions[session_token]
        
        # Check if session has expired
        if datetime.now() > session["expires"]:
            # Clean up expired session
            del self.sessions[session_token]
            return False, None
            
        # Update last activity
        session["last_activity"] = datetime.now()
        
        # Extend session timeout
        session["expires"] = datetime.now() + timedelta(seconds=self.session_timeout)
        
        return True, session
        
    def check_access(self, session_token: str, resource_type: str, resource_name: str) -> bool:
        """
        Check if user has access to a resource.
        
        Args:
            session_token: Session token for authenticated user
            resource_type: Type of resource (function, plugin, etc.)
            resource_name: Name of specific resource
            
        Returns:
            True if access is granted, False otherwise
        """
        if not self.access_control_enabled:
            return True
            
        # Validate session
        valid, session = self.validate_session(session_token)
        if not valid:
            self.log_security_event("access_control", f"Access denied due to invalid session: {resource_type}/{resource_name}")
            return False
            
        role = session["role"]
        
        # Check access rules
        if role not in self.access_rules:
            self.log_security_event("access_control", f"Access denied for unknown role '{role}': {resource_type}/{resource_name}")
            return False
            
        # Get allowed resources for this role and type
        resource_key = f"allowed_{resource_type}s"
        if resource_key not in self.access_rules[role]:
            self.log_security_event("access_control", f"Access denied for role '{role}': {resource_type}/{resource_name} (no rules defined)")
            return False
            
        allowed_resources = self.access_rules[role][resource_key]
        
        # Check for wildcard or specific resource
        has_access = "*" in allowed_resources or resource_name in allowed_resources
        
        if not has_access:
            self.log_security_event("access_control", f"Access denied for role '{role}': {resource_type}/{resource_name}")
        
        return has_access
        
    def log_security_event(self, event_type: str, message: str) -> None:
        """
        Log security event to secure audit log.
        
        Args:
            event_type: Type of security event
            message: Event details
        """
        timestamp = datetime.now().isoformat()
        
        # Format log entry
        log_entry = f"{timestamp} - {event_type} - {message}\n"
        
        try:
            # Append to log file
            with open(self.security_log_path, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Failed to write to security log: {e}")
            
    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        if not self.encryption_enabled:
            return data
            
        try:
            encrypted = self.cipher.encrypt(data.encode('utf-8'))
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            self.log_security_event("encryption", f"Encryption failure: {e}")
            return data
            
    def decrypt_data(self, encrypted_data: str) -> str:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data
        """
        if not self.encryption_enabled:
            return encrypted_data
            
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            self.log_security_event("encryption", f"Decryption failure: {e}")
            return ""
            
    def sanitize_input(self, input_text: str) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            input_text: User input text
            
        Returns:
            Sanitized input text
        """
        # Basic sanitization - could be expanded as needed
        dangerous_patterns = [
            "; rm -rf", "DROP TABLE", "DELETE FROM",
            "<script>", "javascript:", "onerror=",
            "eval(", "exec(", "system(", "import os;"
        ]
        
        sanitized = input_text
        for pattern in dangerous_patterns:
            if pattern.lower() in sanitized.lower():
                self.log_security_event("input_validation", f"Potentially dangerous input blocked: {pattern}")
                sanitized = sanitized.replace(pattern, "[FILTERED]")
                
        return sanitized
        
    def verify_file_integrity(self, file_path: str) -> bool:
        """
        Verify integrity of a file against stored hash.
        
        Args:
            file_path: Path to file to verify
            
        Returns:
            True if integrity verified, False otherwise
        """
        hash_file = Path(f"{file_path}.sha256")
        
        # If no hash file exists, create one
        if not hash_file.exists():
            return self._create_integrity_hash(file_path)
            
        # Calculate current hash
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                current_hash = hashlib.sha256(file_data).hexdigest()
                
            # Compare with stored hash
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
                
            if current_hash != stored_hash:
                self.log_security_event("integrity", f"File integrity check failed: {file_path}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"File integrity check failed: {e}")
            self.log_security_event("integrity", f"File integrity check error: {file_path} - {e}")
            return False
            
    def _create_integrity_hash(self, file_path: str) -> bool:
        """Create integrity hash file for a file."""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                file_hash = hashlib.sha256(file_data).hexdigest()
                
            # Save hash
            with open(f"{file_path}.sha256", 'w') as f:
                f.write(file_hash)
                
            self.log_security_event("integrity", f"Created integrity hash for: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create integrity hash: {e}")
            self.log_security_event("integrity", f"Failed to create integrity hash: {file_path} - {e}")
            return False