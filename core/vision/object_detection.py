"""
Object detection for Sebastian assistant.
"""
import logging
import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from transformers import DetrImageProcessor, DetrForObjectDetection

logger = logging.getLogger(__name__)

class ObjectDetectionEngine:
    """
    Detects and identifies objects in images, providing bounding boxes,
    class labels, and confidence scores.
    """
    
    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        """
        Initialize object detection engine.
        
        Args:
            model_name: Name of the object detection model to use
        """
        try:
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            self.model = DetrForObjectDetection.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"Object detection engine initialized with model {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize object detection: {e}")
            raise
    
    def detect_objects(self, image, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Detect objects in the image.
        
        Args:
            image: NumPy array image (BGR format from OpenCV)
            threshold: Confidence threshold for detection
            
        Returns:
            List of dictionaries containing detection results
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert outputs to COCO API
        target_sizes = torch.tensor([rgb_image.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]  # Convert to Python list of integers
            detection = {
                'label': self.model.config.id2label[label.item()],
                'confidence': score.item(),
                'box': {  # Convert to (x, y, w, h) format
                    'x': box[0],
                    'y': box[1],
                    'width': box[2] - box[0],
                    'height': box[3] - box[1]
                }
            }
            detections.append(detection)
        
        return detections
    
    def analyze_image(self, image) -> Dict[str, Any]:
        """
        Provide a high-level analysis of the image based on detected objects.
        
        Args:
            image: NumPy array image (BGR format from OpenCV)
            
        Returns:
            Dict with analysis results
        """
        detections = self.detect_objects(image)
        
        if not detections:
            return {'summary': 'No objects detected in the image.', 'objects': []}
        
        # Count objects by class
        object_counts = {}
        for det in detections:
            label = det['label']
            object_counts[label] = object_counts.get(label, 0) + 1
        
        # Generate summary
        object_descriptions = []
        for obj, count in object_counts.items():
            if count > 1:
                object_descriptions.append(f"{count} {obj}s")
            else:
                object_descriptions.append(f"{count} {obj}")
        
        if len(object_descriptions) > 1:
            last_item = object_descriptions.pop()
            summary = f"I can see {', '.join(object_descriptions)} and {last_item}."
        else:
            summary = f"I can see {object_descriptions[0]}."
        
        return {
            'summary': summary,
            'objects': detections,
            'object_counts': object_counts,
            'total_objects': len(detections)
        }

