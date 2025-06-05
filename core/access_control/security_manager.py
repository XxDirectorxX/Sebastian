"""
Security manager for Sebastian assistant.

Provides authentication, authorization, secure storage, and
cryptographic operations for the system.
"""
import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key, load_pem_public_key,
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Handles all security-related functionality:
    - User authentication
    - Authorization checks
    - Secure data storage
    - Encryption/decryption
    - Security auditing
    """
    
    def __init__(self, config_dir: str = "core/config/security"):
        """
        Initialize security manager.
        
        Args:
            config_dir: Directory for security configuration
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Key paths
        self.key_path = self.config_dir / "encryption_key.key"
        self.private_key_path = self.config_dir / "private_key.pem"
        self.public_key_path = self.config_dir / "public_key.pem"
        
        # Authorization levels
        self.roles = {
            "administrator": 100,
            "master": 90,
            "family": 70,
            "guest": 50,
            "limited": 30,
            "unidentified": 10
        }
        
        # Load or create encryption keys
        self._init_encryption()
        
        # User database
        self.user_db_path = self.config_dir / "users.json"
        self.user_db = self._load_user_db()
        
        # Session management
        self.sessions = {}
        self.token_secret = self._load_or_create_token_secret()
        
        # Audit logging
        self.audit_log_path = Path("logs/security_audit.log")
        self.audit_log_path.parent.mkdir(exist_ok=True)
        
        logger.info("Security manager initialized")
    
    def _init_encryption(self):
        """Initialize encryption keys."""
        # Symmetric encryption key for data
        if not self.key_path.exists():
            key = Fernet.generate_key()
            with open(self.key_path, 'wb') as f:
                f.write(key)
            logger.info("Generated new encryption key")
        
        with open(self.key_path, 'rb') as f:
            self.encryption_key = f.read()
        
        self.cipher = Fernet(self.encryption_key)
        
        # Asymmetric key pair for signing
        if not self.private_key_path.exists() or not self.public_key_path.exists():
            self._generate_key_pair()
        
        # Load key pair
        with open(self.private_key_path, 'rb') as f:
            self.private_key = load_pem_private_key(f.read(), password=None)
        
        with open(self.public_key_path, 'rb') as f:
            self.public_key = load_pem_public_key(f.read())
        
        logger.info("Encryption initialized")
    
    def _generate_key_pair(self):
        """Generate RSA key pair for signing."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        public_key = private_key.public_key()
        
        # Save private key
        with open(self.private_key_path, 'wb') as f:
            f.write(private_key.private_bytes(
                Encoding.PEM,
                PrivateFormat.PKCS8,
                NoEncryption()
            ))
        
        # Save public key
        with open(self.public_key_path, 'wb') as f:
            f.write(public_key.public_bytes(
                Encoding.PEM,
                PublicFormat.SubjectPublicKeyInfo
            ))
        
        logger.info("Generated new RSA key pair")
    
    def _load_or_create_token_secret(self) -> bytes:
        """Load or create secret for token signing."""
        secret_path = self.config_dir / "token_secret.key"
        
        if not secret_path.exists():
            # Generate a secure random secret
            secret = secrets.token_bytes(32)
            with open(secret_path, 'wb') as f:
                f.write(secret)
            logger.info("Generated new token secret")
            return secret
        
        with open(secret_path, 'rb') as f:
            return f.read()
    
    def _load_user_db(self) -> Dict[str, Any]:
        """Load user database from file."""
        if not self.user_db_path.exists():
            # Create default admin user
            admin_password = secrets.token_urlsafe(16)
            admin_user = {
                "username": "admin",
                "password_hash": self._hash_password(admin_password),
                "role": "administrator",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "face_id": None,
                "voice_id": None,
                "mfa_enabled": False,
                "access_points": ["all"]
            }
            
            # Create default master user
            master_password = secrets.token_urlsafe(16)
            master_user = {
                "username": "master",
                "password_hash": self._hash_password(master_password),
                "role": "master",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "face_id": None,
                "voice_id": None,
                "mfa_enabled": False,
                "access_points": ["all"]
            }
            
            user_db = {
                "users": {
                    "admin": admin_user,
                    "master": master_user
                },
                "last_updated": datetime.now().isoformat()
            }
            
            self._save_user_db(user_db)
            
            # Log the generated credentials (in production, use a secure channel)
            logger.warning(f"Generated default admin credentials: admin/{admin_password}")
            logger.warning(f"Generated default master credentials: master/{master_password}")
            logger.warning("Please change these credentials immediately!")
            
            return user_db
        
        try:
            with open(self.user_db_path, 'r') as f:
                import json
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user database: {e}")
            return {"users": {}, "last_updated": datetime.now().isoformat()}
    
    def _save_user_db(self, user_db: Dict[str, Any]) -> bool:
        """Save user database to file."""
        try:
            user_db["last_updated"] = datetime.now().isoformat()
            with open(self.user_db_path, 'w') as f:
                import json
                json.dump(user_db, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving user database: {e}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        password_bytes = password.encode('utf-8')
        hash_bytes = password_hash.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hash_bytes)
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data using symmetric encryption.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Encrypted data as bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using symmetric encryption.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data as bytes
        """
        return self.cipher.decrypt(encrypted_data)
    
    def sign_data(self, data: Union[str, bytes]) -> bytes:
        """
        Sign data using private key.
        
        Args:
            data: Data to sign
            
        Returns:
            Signature
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, data: Union[str, bytes], signature: bytes) -> bool:
        """
        Verify signature using public key.
        
        Args:
            data: Original data
            signature: Signature to verify
            
        Returns:
            True if signature is valid
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User info dict if authenticated, None otherwise
        """
        users = self.user_db.get("users", {})
        
        if username not in users:
            self.audit_log("AUTH_FAIL", f"Login attempt for non-existent user {username}")
            return None
            
        user = users[username]
        
        if self._verify_password(password, user["password_hash"]):
            # Update last login
            user["last_login"] = datetime.now().isoformat()
            self._save_user_db(self.user_db)
            
            self.audit_log("AUTH_SUCCESS", f"User {username} authenticated")
            return user
        
        self.audit_log("AUTH_FAIL", f"Failed login attempt for user {username}")
        return None
    
    def create_session(self, user_info: Dict[str, Any]) -> str:
        """
        Create a session token for authenticated user.
        
        Args:
            user_info: User information dictionary
            
        Returns:
            Session token
        """
        username = user_info["username"]
        role = user_info["role"]
        
        # Create token
        now = datetime.utcnow()
        expiry = now + timedelta(hours=12)  # 12 hour session
        
        payload = {
            "sub": username,
            "role": role,
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp()),
            "jti": secrets.token_hex(16)  # Unique token ID
        }
        
        token = jwt.encode(payload, self.token_secret, algorithm="HS256")
        
        # Store session
        session_id = payload["jti"]
        self.sessions[session_id] = {
            "username": username,
            "created_at": now.isoformat(),
            "expires_at": expiry.isoformat(),
            "role": role,
            "last_activity": now.isoformat()
        }
        
        self.audit_log("SESSION_CREATE", f"Session created for user {username}")
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session token.
        
        Args:
            token: Session token
            
        Returns:
            Session info if valid, None otherwise
        """
        try:
            # Decode and verify token
            payload = jwt.decode(token, self.token_secret, algorithms=["HS256"])
            
            session_id = payload["jti"]
            
            # Check if session exists
            if session_id not in self.sessions:
                self.audit_log("SESSION_INVALID", f"Token for non-existent session {session_id}")
                return None
                
            # Update last activity
            self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
            
            return self.sessions[session_id]
            
        except jwt.ExpiredSignatureError:
            self.audit_log("SESSION_EXPIRED", "Expired token used")
            return None
        except jwt.InvalidTokenError:
            self.audit_log("SESSION_INVALID", "Invalid token used")
            return None
    
    def end_session(self, token: str) -> bool:
        """
        End a user session.
        
        Args:
            token: Session token
            
        Returns:
            Success flag
        """
        try:
            payload = jwt.decode(token, self.token_secret, algorithms=["HS256"])
            session_id = payload["jti"]
            
            if session_id in self.sessions:
                username = self.sessions[session_id]["username"]
                del self.sessions[session_id]
                self.audit_log("SESSION_END", f"Session ended for user {username}")
                return True
                
            return False
            
        except jwt.InvalidTokenError:
            return False
    
    def check_authorization(self, token: str, required_role: str) -> bool:
        """
        Check if user has required role.
        
        Args:
            token: Session token
            required_role: Required role
            
        Returns:
            True if authorized
        """
        session = self.validate_token(token)
        if not session:
            return False
            
        user_role = session["role"]
        user_level = self.roles.get(user_role, 0)
        required_level = self.roles.get(required_role, 100)  # Default to highest level
        
        is_authorized = user_level >= required_level
        
        if not is_authorized:
            self.audit_log(
                "AUTH_DENIED", 
                f"User {session['username']} ({user_role}) denied access requiring {required_role}"
            )
            
        return is_authorized
    
    def add_user(self, admin_token: str, user_data: Dict[str, Any]) -> bool:
        """
        Add a new user (requires administrator token).
        
        Args:
            admin_token: Administrator session token
            user_data: New user data
            
        Returns:
            Success flag
        """
        if not self.check_authorization(admin_token, "administrator"):
            return False
            
        username = user_data.get("username")
        password = user_data.get("password")
        role = user_data.get("role", "guest")
        
        if not username or not password or username in self.user_db.get("users", {}):
            return False
            
        # Create user
        user = {
            "username": username,
            "password_hash": self._hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "face_id": user_data.get("face_id"),
            "voice_id": user_data.get("voice_id"),
            "mfa_enabled": user_data.get("mfa_enabled", False),
            "access_points": user_data.get("access_points# filepath: c:\Users\Iam\Desktop\Sebastian\core\access_control\security_manager.py
"""
Security manager for Sebastian assistant.

Provides authentication, authorization, secure storage, and
cryptographic operations for the system.
"""
import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key, load_pem_public_key,
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Handles all security-related functionality:
    - User authentication
    - Authorization checks
    - Secure data storage
    - Encryption/decryption
    - Security auditing
    """
    
    def __init__(self, config_dir: str = "core/config/security"):
        """
        Initialize security manager.
        
        Args:
            config_dir: Directory for security configuration
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Key paths
        self.key_path = self.config_dir / "encryption_key.key"
        self.private_key_path = self.config_dir / "private_key.pem"
        self.public_key_path = self.config_dir / "public_key.pem"
        
        # Authorization levels
        self.roles = {
            "administrator": 100,
            "master": 90,
            "family": 70,
            "guest": 50,
            "limited": 30,
            "unidentified": 10
        }
        
        # Load or create encryption keys
        self._init_encryption()
        
        # User database
        self.user_db_path = self.config_dir / "users.json"
        self.user_db = self._load_user_db()
        
        # Session management
        self.sessions = {}
        self.token_secret = self._load_or_create_token_secret()
        
        # Audit logging
        self.audit_log_path = Path("logs/security_audit.log")
        self.audit_log_path.parent.mkdir(exist_ok=True)
        
        logger.info("Security manager initialized")
    
    def _init_encryption(self):
        """Initialize encryption keys."""
        # Symmetric encryption key for data
        if not self.key_path.exists():
            key = Fernet.generate_key()
            with open(self.key_path, 'wb') as f:
                f.write(key)
            logger.info("Generated new encryption key")
        
        with open(self.key_path, 'rb') as f:
            self.encryption_key = f.read()
        
        self.cipher = Fernet(self.encryption_key)
        
        # Asymmetric key pair for signing
        if not self.private_key_path.exists() or not self.public_key_path.exists():
            self._generate_key_pair()
        
        # Load key pair
        with open(self.private_key_path, 'rb') as f:
            self.private_key = load_pem_private_key(f.read(), password=None)
        
        with open(self.public_key_path, 'rb') as f:
            self.public_key = load_pem_public_key(f.read())
        
        logger.info("Encryption initialized")
    
    def _generate_key_pair(self):
        """Generate RSA key pair for signing."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        public_key = private_key.public_key()
        
        # Save private key
        with open(self.private_key_path, 'wb') as f:
            f.write(private_key.private_bytes(
                Encoding.PEM,
                PrivateFormat.PKCS8,
                NoEncryption()
            ))
        
        # Save public key
        with open(self.public_key_path, 'wb') as f:
            f.write(public_key.public_bytes(
                Encoding.PEM,
                PublicFormat.SubjectPublicKeyInfo
            ))
        
        logger.info("Generated new RSA key pair")
    
    def _load_or_create_token_secret(self) -> bytes:
        """Load or create secret for token signing."""
        secret_path = self.config_dir / "token_secret.key"
        
        if not secret_path.exists():
            # Generate a secure random secret
            secret = secrets.token_bytes(32)
            with open(secret_path, 'wb') as f:
                f.write(secret)
            logger.info("Generated new token secret")
            return secret
        
        with open(secret_path, 'rb') as f:
            return f.read()
    
    def _load_user_db(self) -> Dict[str, Any]:
        """Load user database from file."""
        if not self.user_db_path.exists():
            # Create default admin user
            admin_password = secrets.token_urlsafe(16)
            admin_user = {
                "username": "admin",
                "password_hash": self._hash_password(admin_password),
                "role": "administrator",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "face_id": None,
                "voice_id": None,
                "mfa_enabled": False,
                "access_points": ["all"]
            }
            
            # Create default master user
            master_password = secrets.token_urlsafe(16)
            master_user = {
                "username": "master",
                "password_hash": self._hash_password(master_password),
                "role": "master",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "face_id": None,
                "voice_id": None,
                "mfa_enabled": False,
                "access_points": ["all"]
            }
            
            user_db = {
                "users": {
                    "admin": admin_user,
                    "master": master_user
                },
                "last_updated": datetime.now().isoformat()
            }
            
            self._save_user_db(user_db)
            
            # Log the generated credentials (in production, use a secure channel)
            logger.warning(f"Generated default admin credentials: admin/{admin_password}")
            logger.warning(f"Generated default master credentials: master/{master_password}")
            logger.warning("Please change these credentials immediately!")
            
            return user_db
        
        try:
            with open(self.user_db_path, 'r') as f:
                import json
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user database: {e}")
            return {"users": {}, "last_updated": datetime.now().isoformat()}
    
    def _save_user_db(self, user_db: Dict[str, Any]) -> bool:
        """Save user database to file."""
        try:
            user_db["last_updated"] = datetime.now().isoformat()
            with open(self.user_db_path, 'w') as f:
                import json
                json.dump(user_db, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving user database: {e}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        password_bytes = password.encode('utf-8')
        hash_bytes = password_hash.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hash_bytes)
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data using symmetric encryption.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Encrypted data as bytes
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using symmetric encryption.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data as bytes
        """
        return self.cipher.decrypt(encrypted_data)
    
    def sign_data(self, data: Union[str, bytes]) -> bytes:
        """
        Sign data using private key.
        
        Args:
            data: Data to sign
            
        Returns:
            Signature
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, data: Union[str, bytes], signature: bytes) -> bool:
        """
        Verify signature using public key.
        
        Args:
            data: Original data
            signature: Signature to verify
            
        Returns:
            True if signature is valid
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User info dict if authenticated, None otherwise
        """
        users = self.user_db.get("users", {})
        
        if username not in users:
            self.audit_log("AUTH_FAIL", f"Login attempt for non-existent user {username}")
            return None
            
        user = users[username]
        
        if self._verify_password(password, user["password_hash"]):
            # Update last login
            user["last_login"] = datetime.now().isoformat()
            self._save_user_db(self.user_db)
            
            self.audit_log("AUTH_SUCCESS", f"User {username} authenticated")
            return user
        
        self.audit_log("AUTH_FAIL", f"Failed login attempt for user {username}")
        return None
    
    def create_session(self, user_info: Dict[str, Any]) -> str:
        """
        Create a session token for authenticated user.
        
        Args:
            user_info: User information dictionary
            
        Returns:
            Session token
        """
        username = user_info["username"]
        role = user_info["role"]
        
        # Create token
        now = datetime.utcnow()
        expiry = now + timedelta(hours=12)  # 12 hour session
        
        payload = {
            "sub": username,
            "role": role,
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp()),
            "jti": secrets.token_hex(16)  # Unique token ID
        }
        
        token = jwt.encode(payload, self.token_secret, algorithm="HS256")
        
        # Store session
        session_id = payload["jti"]
        self.sessions[session_id] = {
            "username": username,
            "created_at": now.isoformat(),
            "expires_at": expiry.isoformat(),
            "role": role,
            "last_activity": now.isoformat()
        }
        
        self.audit_log("SESSION_CREATE", f"Session created for user {username}")
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session token.
        
        Args:
            token: Session token
            
        Returns:
            Session info if valid, None otherwise
        """
        try:
            # Decode and verify token
            payload = jwt.decode(token, self.token_secret, algorithms=["HS256"])
            
            session_id = payload["jti"]
            
            # Check if session exists
            if session_id not in self.sessions:
                self.audit_log("SESSION_INVALID", f"Token for non-existent session {session_id}")
                return None
                
            # Update last activity
            self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
            
            return self.sessions[session_id]
            
        except jwt.ExpiredSignatureError:
            self.audit_log("SESSION_EXPIRED", "Expired token used")
            return None
        except jwt.InvalidTokenError:
            self.audit_log("SESSION_INVALID", "Invalid token used")
            return None
    
    def end_session(self, token: str) -> bool:
        """
        End a user session.
        
        Args:
            token: Session token
            
        Returns:
            Success flag
        """
        try:
            payload = jwt.decode(token, self.token_secret, algorithms=["HS256"])
            session_id = payload["jti"]
            
            if session_id in self.sessions:
                username = self.sessions[session_id]["username"]
                del self.sessions[session_id]
                self.audit_log("SESSION_END", f"Session ended for user {username}")
                return True
                
            return False
            
        except jwt.InvalidTokenError:
            return False
    
    def check_authorization(self, token: str, required_role: str) -> bool:
        """
        Check if user has required role.
        
        Args:
            token: Session token
            required_role: Required role
            
        Returns:
            True if authorized
        """
        session = self.validate_token(token)
        if not session:
            return False
            
        user_role = session["role"]
        user_level = self.roles.get(user_role, 0)
        required_level = self.roles.get(required_role, 100)  # Default to highest level
        
        is_authorized = user_level >= required_level
        
        if not is_authorized:
            self.audit_log(
                "AUTH_DENIED", 
                f"User {session['username']} ({user_role}) denied access requiring {required_role}"
            )
            
        return is_authorized
    
    def add_user(self, admin_token: str, user_data: Dict[str, Any]) -> bool:
        """
        Add a new user (requires administrator token).
        
        Args:
            admin_token: Administrator session token
            user_data: New user data
            
        Returns:
            Success flag
        """
        if not self.check_authorization(admin_token, "administrator"):
            return False
            
        username = user_data.get("username")
        password = user_data.get("password")
        role = user_data.get("role", "guest")
        
        if not username or not password or username in self.user_db.get("users", {}):
            return False
            
        # Create user
        user = {
            "username": username,
            "password_hash": self._hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "face_id": user_data.get("face_id"),
            "voice_id": user_data.get("voice_id"),
            "mfa_enabled": user_data.get("mfa_enabled", False),
            "access_points": user_data.get("access_points