"""
vision_manager.py — A Phantomhive-Grade Multimodal Surveillance Core
Sebastian's visual awareness engine for identity scoring, presence detection,
and access control. Camera-agnostic. Modular. Elegant.
"""

import cv2
import threading
import time
import face_recognition
import numpy as np
import os
import logging

from core.vision.face_recognition import FaceRecognitionEngine
from core.vision.object_detection import ObjectDetectionEngine
from core.vision.scene_analysis import SceneAnalysisEngine

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Sebastian.VisionManager")

class CameraFeed:
    def __init__(self, source_url, camera_id, role='general', resolution=(640, 480)):
        self.source_url = source_url
        self.camera_id = camera_id
        self.role = role
        self.capture = None
        self.thread = None
        self.active = False
        self.latest_frame = None
        self.resolution = resolution

    def start(self):
        self.capture = cv2.VideoCapture(self.source_url)
        if not self.capture.isOpened():
            logger.error(f"Camera {self.camera_id} failed to open.")
            return
        self.active = True
        self.thread = threading.Thread(target=self._update_feed, daemon=True)
        self.thread.start()
        logger.info(f"Camera {self.camera_id} feed started.")

    def _update_feed(self):
        while self.active:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.resize(frame, self.resolution)
                self.latest_frame = frame
            time.sleep(0.1)  # 10 FPS throttle

    def read(self):
        return self.latest_frame

    def stop(self):
        self.active = False
        if self.capture:
            self.capture.release()
        logger.info(f"Camera {self.camera_id} feed stopped.")

class VisionManager:
    """
    Manages vision-related capabilities including face recognition,
    object detection, and scene analysis. Coordinates camera access
    and processing pipelines.
    """
    
    def __init__(self):
        """Initialize vision manager and its component engines."""
        self.face_engine = None
        self.object_engine = None
        self.scene_engine = None
        
        # Camera resources
        self.camera = None
        self.camera_id = 0
        self.camera_lock = threading.Lock()
        self.camera_thread = None
        self.is_camera_running = False
        
        # Initialize engines on demand to save resources
        logger.info("Vision manager initialized")
    
    def _lazy_init_face_engine(self):
        """Initialize face recognition engine if not already initialized."""
        if self.face_engine is None:
            logger.info("Initializing face recognition engine")
            self.face_engine = FaceRecognitionEngine()
    
    def _lazy_init_object_engine(self):
        """Initialize object detection engine if not already initialized."""
        if self.object_engine is None:
            logger.info("Initializing object detection engine")
            self.object_engine = ObjectDetectionEngine()
    
    def _lazy_init_scene_engine(self):
        """Initialize scene analysis engine if not already initialized."""
        if self.scene_engine is None:
            logger.info("Initializing scene analysis engine")
            self.scene_engine = SceneAnalysisEngine()
    
    def _open_camera(self) -> bool:
        """Open camera if not already open."""
        with self.camera_lock:
            if self.camera is None:
                self.camera = cv2.VideoCapture(self.camera_id)
                if not self.camera.isOpened():
                    logger.error(f"Failed to open camera {self.camera_id}")
                    self.camera = None
                    return False
                logger.info(f"Opened camera {self.camera_id}")
            return self.camera.isOpened()
    
    def _close_camera(self):
        """Close the camera if it's open."""
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                logger.info("Camera closed")
    
    def _get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture a frame from the camera."""
        if not self._open_camera():
            return False, None
        
        with self.camera_lock:
            ret, frame = self.camera.read()
        
        if not ret:
            logger.warning("Failed to capture frame")
            return False, None
            
        return True, frame
    
    def detect_faces(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Detect and recognize faces using the camera.
        
        Returns:
            Tuple of success flag and list of face detection results
        """
        self._lazy_init_face_engine()
        
        success, frame = self._get_frame()
        if not success:
            return False, []
        
        try:
            faces = self.face_engine.recognize_faces(frame)
            return True, faces
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return False, []
    
    def register_face(self, name: str) -> bool:
        """
        Register a face with the given name.
        
        Args:
            name: Name to associate with the face
            
        Returns:
            bool: Success or failure
        """
        self._lazy_init_face_engine()
        
        # Capture multiple frames to find the best one
        best_frame = None
        best_face_size = 0
        
        for _ in range(5):  # Try 5 frames
            success, frame = self._get_frame()
            if not success:
                continue
            
            # Detect face and find the largest one
            face_locations = self.face_engine.detect_faces(frame)
            if face_locations:
                for face_loc in face_locations:
                    # Calculate face size
                    face_height = face_loc[2] - face_loc[0]
                    face_width = face_loc[1] - face_loc[3]
                    face_size = face_height * face_width
                    
                    if face_size > best_face_size:
                        best_frame = frame
                        best_face_size = face_size
        
        if best_frame is None:
            logger.warning("No suitable face found for registration")
            return False
        
        # Register the face
        return self.face_engine.register_face(best_frame, name)
    
    def detect_objects(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect objects in the current camera view.
        
        Returns:
            Tuple of success flag and detection results
        """
        self._lazy_init_object_engine()
        
        success, frame = self._get_frame()
        if not success:
            return False, {}
        
        try:
            results = self.object_engine.analyze_image(frame)
            return True, results
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return False, {}
    
    def analyze_scene(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze the scene in the current camera view.
        
        Returns:
            Tuple of success flag and analysis results
        """
        self._lazy_init_scene_engine()
        
        success, frame = self._get_frame()
        if not success:
            return False, {}
        
        try:
            results = self.scene_engine.analyze_image(frame)
            return True, results
        except Exception as e:
            logger.error(f"Error analyzing scene: {e}")
            return False, {}
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis including faces, objects, and scene.
        
        Returns:
            Dict containing all analysis results
        """
        success, frame = self._get_frame()
        if not success:
            return {'error': 'Failed to capture image'}
        
        results = {}
        
        # Initialize all engines
        self._lazy_init_face_engine()
        self._lazy_init_object_engine()
        self._lazy_init_scene_engine()
        
        # Perform analysis
        try:
            results['faces'] = self.face_engine.recognize_faces(frame)
        except Exception as e:
            logger.error(f"Face analysis error: {e}")
            results['faces'] = {'error': str(e)}
        
        try:
            results['objects'] = self.object_engine.analyze_image(frame)
        except Exception as e:
            logger.error(f"Object analysis error: {e}")
            results['objects'] = {'error': str(e)}
        
        try:
            results['scene'] = self.scene_engine.analyze_image(frame)
        except Exception as e:
            logger.error(f"Scene analysis error: {e}")
            results['scene'] = {'error': str(e)}
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        self._close_camera()
        logger.info("Vision manager resources cleaned up")

    def add_camera(self, source_url, camera_id, role='general'):
        feed = CameraFeed(source_url, camera_id, role)
        self.feeds[camera_id] = feed
        feed.start()

    def remove_camera(self, camera_id):
        if camera_id in self.feeds:
            self.feeds[camera_id].stop()
            del self.feeds[camera_id]

    def identify_faces(self):
        identities = []
        for feed in self.feeds.values():
            frame = feed.read()
            if frame is None:
                continue
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_encodings, encoding, self.tolerance)
                name = "Unknown"
                if True in matches:
                    matched_idx = np.argmin(face_recognition.face_distance(self.known_encodings, encoding))
                    name = self.known_names[matched_idx]
                identities.append(name)
        return identities

    def shutdown(self):
        for feed in self.feeds.values():
            feed.stop()
        logger.info("All camera feeds terminated.")

# Example usage (would be in main assistant core):
# vision = VisionManager()
# vision.add_camera(0, "usb_cam")
# identities = vision.identify_faces()
# print("Detected:", identities)
# vision.shutdown()
