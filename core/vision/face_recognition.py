"""
Face recognition for Sebastian assistant.
"""
import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2
import face_recognition

logger = logging.getLogger(__name__)

class FaceRecognitionEngine:
    """
    Provides face detection, recognition and authentication capabilities.
    Uses face_recognition library (dlib-based) for high-accuracy recognition.
    """
    
    def __init__(self, data_path: str = "assets/face_data/user_faces.dat"):
        """
        Initialize face recognition engine.
        
        Args:
            data_path: Path to saved face encodings
        """
        self.data_path = data_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.recognition_tolerance = 0.6  # Lower = stricter matching
        
        # Load known faces if available
        self._load_known_faces()
        logger.info(f"Face recognition engine initialized with {len(self.known_face_names)} known faces")
    
    def _load_known_faces(self):
        """Load known face encodings from disk."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                logger.info(f"Loaded {len(self.known_face_names)} faces from {self.data_path}")
            except Exception as e:
                logger.error(f"Error loading face data: {e}")
    
    def _save_known_faces(self):
        """Save known face encodings to disk."""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        try:
            with open(self.data_path, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            logger.info(f"Saved {len(self.known_face_names)} faces to {self.data_path}")
        except Exception as e:
            logger.error(f"Error saving face data: {e}")
    
    def detect_faces(self, image) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image without recognition.
        
        Args:
            image: NumPy array image (BGR format from OpenCV)
            
        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        return face_locations
    
    def recognize_faces(self, image) -> List[Dict[str, Any]]:
        """
        Detect and recognize faces in image.
        
        Args:
            image: NumPy array image (BGR format from OpenCV)
            
        Returns:
            List of dictionaries with 'name', 'confidence', 'location' keys
        """
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        
        # If no faces or no known encodings, return empty results
        if not face_locations or not self.known_face_encodings:
            return []
        
        # Generate encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.recognition_tolerance
            )
            
            name = "Unknown"
            confidence = 0.0
            
            if True in matches:
                # Calculate face distances to find best match
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                confidence = 1.0 - min(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            results.append({
                'name': name,
                'confidence': float(confidence),  # Convert from numpy float to Python float
                'location': face_location
            })
        
        return results
    
    def register_face(self, image, name: str) -> bool:
        """
        Register a new face or update existing face.
        
        Args:
            image: NumPy array image (BGR format from OpenCV)
            name: Name of the person
            
        Returns:
            bool: Success or failure
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces - expecting only one face during registration
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            logger.warning("No face found in registration image")
            return False
        
        if len(face_locations) > 1:
            logger.warning(f"Multiple faces found in registration image. Using the first one.")
        
        # Generate encoding for the face
        face_encodings = face_recognition.face_encodings(rgb_image, [face_locations[0]])
        
        if not face_encodings:
            logger.error("Failed to generate face encoding")
            return False
        
        # Check if name already exists
        if name in self.known_face_names:
            idx = self.known_face_names.index(name)
            # Update existing encoding
            self.known_face_encodings[idx] = face_encodings[0]
            logger.info(f"Updated face encoding for {name}")
        else:
            # Add new face
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            logger.info(f"Registered new face for {name}")
        
        # Save updated face data
        self._save_known_faces()
        return True
    
    def delete_face(self, name: str) -> bool:
        """
        Delete a registered face.
        
        Args:
            name: Name of the person to delete
            
        Returns:
            bool: Success or failure
        """
        if name in self.known_face_names:
            idx = self.known_face_names.index(name)
            self.known_face_encodings.pop(idx)
            self.known_face_names.pop(idx)
            self._save_known_faces()
            logger.info(f"Deleted face data for {name}")
            return True
        else:
            logger.warning(f"No face data found for {name}")
            return False

