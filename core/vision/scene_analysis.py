"""
Scene analysis for Sebastian assistant.
"""
import logging
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

class SceneAnalysisEngine:
    """
    Analyzes images to extract high-level information about scenes, activities, 
    and general content using computer vision techniques.
    """
    
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        """
        Initialize scene analysis engine.
        
        Args:
            model_name: Name of the image classification model to use
        """
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            logger.info(f"Scene analysis engine initialized with model {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize scene analysis: {e}")
            # Fallback to a simpler model or raise error depending on requirements
            raise
    
    def analyze_image(self, image) -> Dict[str, Any]:
        """
        Analyze image to identify scene, objects, and activities.
        
        Args:
            image: NumPy array image (BGR format from OpenCV)
            
        Returns:
            Dict containing scene analysis results
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Process image through model
        inputs = self.processor(images=pil_image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get predicted class and confidence
        predicted_class_idx = logits.argmax(-1).item()
        confidence = logits.softmax(dim=-1)[0, predicted_class_idx].item()
        
        # Get class label
        predicted_label = self.model.config.id2label[predicted_class_idx]
        
        # Get top-5 predictions
        top_indices = logits.softmax(dim=-1)[0].topk(5).indices
        top_predictions = [
            {
                'label': self.model.config.id2label[idx.item()],
                'confidence': logits.softmax(dim=-1)[0, idx].item()
            }
            for idx in top_indices
        ]
        
        # Basic color analysis
        color_analysis = self._analyze_colors(rgb_image)
        
        # Brightness and contrast
        brightness = np.mean(rgb_image)
        
        return {
            'scene': predicted_label,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'colors': color_analysis,
            'brightness': brightness
        }
    
    def _analyze_colors(self, image) -> Dict[str, float]:
        """
        Analyze dominant colors in the image.
        
        Args:
            image: RGB image as NumPy array
            
        Returns:
            Dict with color analysis results
        """
        # Reshape image and convert to float
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Number of clusters for color quantization
        n_colors = 5
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count labels to find dominant colors
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        centers = centers[sorted_indices]
        counts = counts[sorted_indices]
        
        # Calculate percentage of each color
        total_pixels = image.shape[0] * image.shape[1]
        color_percentages = counts / total_pixels
        
        result = {}
        for i in range(min(3, len(centers))):  # Return top 3 colors
            color = tuple(int(c) for c in centers[i])
            result[f"color_{i+1}"] = {
                'rgb': color,
                'hex': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                'percentage': float(color_percentages[i])
            }
        
        return result

