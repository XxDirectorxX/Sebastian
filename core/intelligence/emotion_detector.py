"""
Multimodal emotion analysis for Sebastian assistant.

Detects emotions from various input modalities including text,
voice acoustics, and facial expressions, then provides a unified
emotional assessment.
"""
import logging
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import asyncio
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class EmotionDetector:
    """
    Multimodal emotion detection system.
    
    Analyzes emotions from text content, voice characteristics,
    facial expressions, and contextual cues, then provides a
    unified emotional assessment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize emotion detector with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Emotion categories
        self.emotion_categories = [
            "neutral", "happy", "sad", "angry", "fearful", 
            "disgusted", "surprised", "confused", "interested"
        ]
        
        # Confidence thresholds
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        
        # Integration weights for different modalities
        self.modality_weights = {
            "text": self.config.get("text_weight", 0.4),
            "voice": self.config.get("voice_weight", 0.3),
            "face": self.config.get("face_weight", 0.3),
            "context": self.config.get("context_weight", 0.2)
        }
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(self.modality_weights.values())
        for key in self.modality_weights:
            self.modality_weights[key] /= weight_sum
            
        # Load models
        self._load_models()
        
        logger.info("Emotion detector initialized")
        
    def _load_models(self):
        """Load emotion detection models for different modalities."""
        # Initialize flags for available models
        self.text_model_available = False
        self.voice_model_available = False  
        self.face_model_available = False
        
        # Load text emotion model
        try:
            self._load_text_emotion_model()
            self.text_model_available = True
        except Exception as e:
            logger.warning(f"Failed to load text emotion model: {e}")
            
        # Load voice emotion model
        try:
            self._load_voice_emotion_model()
            self.voice_model_available = True
        except Exception as e:
            logger.warning(f"Failed to load voice emotion model: {e}")
            
        # Load facial emotion model
        try:
            self._load_face_emotion_model()  
            self.face_model_available = True
        except Exception as e:
            logger.warning(f"Failed to load facial emotion model: {e}")
            
    def _load_text_emotion_model(self):
        """Load model for text emotion analysis."""
        model_path = self.config.get("text_model_path", "")
        
        try:
            # Try to load transformer-based sentiment model
            from transformers.pipelines import pipeline
            
            # Select appropriate model based on availability
            if model_path and os.path.exists(model_path):
                self.text_emotion_model = pipeline(
                    "text-classification", 
                    model=model_path,
                    top_k=None
                )
            else:
                # Default to standard emotion model
                self.text_emotion_model = pipeline(
                    "text-classification", 
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None
                )
                
            logger.info("Text emotion model loaded successfully")
            
        except ImportError:
            logger.warning("Transformers library not available. Text emotion detection will be limited.")
            # Fallback to basic sentiment analysis
            import nltk
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
                
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self.text_emotion_model = SentimentIntensityAnalyzer()
            
            # Flag that we're using the fallback model
            self.using_fallback_text_model = True
            logger.info("Using fallback text sentiment analysis")
            
    def _load_voice_emotion_model(self):
        """Load model for voice emotion analysis."""
        model_path = self.config.get("voice_model_path", "")
        
        try:
            # Try to import librosa for audio feature extraction
            import librosa  # noqa
            self.librosa_available = True
            
            # Load appropriate model based on availability
            if model_path and os.path.exists(model_path):
                # Custom model
                import torch
                self.voice_emotion_model = torch.load(model_path)
                self.voice_emotion_model.eval()
            else:
                # Use integrated feature-based classifier
                from sklearn.ensemble import RandomForestClassifier
                self.voice_emotion_model = self._create_audio_feature_classifier()
                
            logger.info("Voice emotion model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Required library not available for voice emotion: {e}")
            self.librosa_available = False
            
    def _create_audio_feature_classifier(self):
        """Create a simple audio feature classifier as fallback."""
        # This would normally load a pre-trained model 
        # For now, return a simple placeholder
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # In a real implementation, this would be pre-trained
        # Here we'll just note that it needs training
        self.voice_model_trained = False
        logger.warning("Voice emotion model requires training")
        
        return model
        
    def _load_face_emotion_model(self):
        """Load model for facial emotion analysis."""
        model_path = self.config.get("face_model_path", "")
        
        try:
            # Try to import face analysis libraries
            import cv2
            self.cv2_available = True
            
            # Load face detection first
            face_cascade_path = self.config.get(
                "face_cascade_path", 
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            # Load emotion detection model
            if model_path and os.path.exists(model_path):
                # Use specified model - implementation depends on model type
                if model_path.endswith('.xml'):
                    # OpenCV DNN model
                    self.face_emotion_model = cv2.dnn.readNet(model_path)
                elif model_path.endswith('.h5'):
                    # Keras model
                    from tensorflow.keras.models import load_model
                    self.face_emotion_model = load_model(model_path)
                else:
                    raise ValueError(f"Unsupported face model format: {model_path}")
            else:
                # Use basic HOG + SVM classifier as fallback
                # This is a placeholder - would be pre-trained in a real implementation
                from sklearn.svm import SVC
                self.face_emotion_model = SVC(kernel='linear', probability=True)
                self.face_model_trained = False
                logger.warning("Face emotion model requires training")
                
            logger.info("Face emotion model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Required library not available for face emotion: {e}")
            self.cv2_available = False
            
    async def detect_emotion(self, 
                           text: Optional[str] = None,
                           audio_data: Optional[Union[np.ndarray, bytes, str]] = None,
                           image_data: Optional[Union[np.ndarray, bytes, str]] = None,
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect emotion from multimodal inputs asynchronously.
        
        Args:
            text: Optional text to analyze
            audio_data: Optional audio data (numpy array, bytes, or file path)
            image_data: Optional image data (numpy array, bytes, or file path)
            context: Optional contextual information
            
        Returns:
            Dictionary with emotion analysis results
        """
        tasks = []
        results = {}
        
        # Process available modalities concurrently
        if text:
            tasks.append(self._detect_text_emotion_async(text))
            
        if audio_data is not None and self.voice_model_available:
            tasks.append(self._detect_voice_emotion_async(audio_data))
            
        if image_data is not None and self.face_model_available:
            tasks.append(self._detect_face_emotion_async(image_data))
            
        # Get contextual emotion if available
        context_emotion = None
        if context and "emotion" in context:
            context_emotion = self._format_context_emotion(context)
            
        # Wait for all detection tasks to complete
        if tasks:
            modality_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results, ignoring exceptions
            for result in modality_results:
                if isinstance(result, dict) and "modality" in result:
                    results[result["modality"]] = result
                elif isinstance(result, Exception):
                    logger.error(f"Emotion detection error: {result}")
                    
        # Add context emotion if available
        if context_emotion:
            results["context"] = context_emotion
            
        # Combine results from all modalities
        combined_result = self._combine_emotions(results, context)
        
        return combined_result
        
    async def _detect_text_emotion_async(self, text: str) -> Dict[str, Any]:
        """Detect emotion from text asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.detect_text_emotion(text)
        )
        
    async def _detect_voice_emotion_async(self, audio_data: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """Detect emotion from voice asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.detect_voice_emotion(audio_data)
        )
        
    async def _detect_face_emotion_async(self, image_data: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """Detect emotion from facial expression asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.detect_face_emotion(image_data)
        )
        
    def detect_text_emotion(self, text: str) -> Dict[str, Any]:
        """
        Detect emotion from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text emotion analysis
        """
        if not text or not self.text_model_available:
            return {"modality": "text", "error": "Text analysis unavailable"}
            
        start_time = time.time()
        
        try:
            # Check if we're using the fallback model (VADER)
            if hasattr(self, 'using_fallback_text_model') and self.using_fallback_text_model:
                # VADER sentiment analysis
                sentiment_scores = self.text_emotion_model.polarity_scores(text)
                
                # Map VADER sentiment to emotions
                emotions = {
                    "neutral": sentiment_scores["neu"],
                    "happy": sentiment_scores["pos"] * 0.8,
                    "angry": sentiment_scores["neg"] * 0.5,
                    "sad": sentiment_scores["neg"] * 0.5,
                    "fearful": 0.0,
                    "disgusted": 0.0,
                    "surprised": 0.0,
                    "confused": 0.0,
                    "interested": 0.0
                }
                
                # Determine dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
            else:
                # Use transformer-based emotion model
                result = self.text_emotion_model(text)
                
                # Format results
                emotions = {label["label"].lower(): label["score"] for label in result[0]}
                
                # Fill in missing emotions with zeros
                for emotion in self.emotion_categories:
                    if emotion not in emotions:
                        emotions[emotion] = 0.0
                        
                # Determine dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
            process_time = time.time() - start_time
            
            return {
                "modality": "text",
                "emotions": emotions,
                "dominant_emotion": dominant_emotion[0],
                "confidence": dominant_emotion[1],
                "process_time": process_time
            }
            
        except Exception as e:
            logger.error(f"Text emotion detection error: {e}", exc_info=True)
            return {
                "modality": "text",
                "error": str(e),
                "emotions": {emotion: 0.0 for emotion in self.emotion_categories},
                "dominant_emotion": "neutral",
                "confidence": 0.0
            }
            
    def detect_voice_emotion(self, audio_data: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """
        Detect emotion from voice audio.
        
        Args:
            audio_data: Audio as numpy array, bytes, or file path
            
        Returns:
            Dictionary with voice emotion analysis
        """
        if not self.voice_model_available or not self.librosa_available:
            return {"modality": "voice", "error": "Voice analysis unavailable"}
            
        start_time = time.time()
        
        try:
            # Process audio data to the right format
            if isinstance(audio_data, bytes):
                # Save bytes to temp file and load with librosa
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
                    temp_path = temp.name
                    temp.write(audio_data)
                
                import librosa
                audio_array, sample_rate = librosa.load(temp_path, sr=None)
                os.unlink(temp_path)
                
            elif isinstance(audio_data, str):
                # Load audio file
                import librosa
                audio_array, sample_rate = librosa.load(audio_data, sr=None)
                
            else:
                # Assume numpy array
                audio_array = audio_data
                sample_rate = 16000  # Default sample rate
                
            # Extract audio features
            features = self._extract_audio_features(audio_array, sample_rate)
            
            # Check if model is trained
            if hasattr(self, 'voice_model_trained') and not self.voice_model_trained:
                # Model not trained, return placeholder
                emotions = {
                    "neutral": 0.7,
                    "happy": 0.1,
                    "sad": 0.05,
                    "angry": 0.05,
                    "fearful": 0.03,
                    "disgusted": 0.02,
                    "surprised": 0.03,
                    "confused": 0.01,
                    "interested": 0.01
                }
            else:
                # Use model to predict emotions
                # Implementation depends on model type
                if hasattr(self.voice_emotion_model, 'predict_proba'):
                    # Scikit-learn style model
                    probabilities = self.voice_emotion_model.predict_proba([features])[0]
                    emotions = {
                        emotion: float(prob) 
                        for emotion, prob in zip(self.emotion_categories, probabilities)
                    }
                else:
                    # Assume PyTorch model
                    import torch
                    with torch.no_grad():
                        tensor_features = torch.tensor(features, dtype=torch.float32)
                        outputs = self.voice_emotion_model(tensor_features.unsqueeze(0))
                        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                        emotions = {
                            emotion: float(prob) 
                            for emotion, prob in zip(self.emotion_categories, probabilities)
                        }
            
            # Determine dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            process_time = time.time() - start_time
            
            return {
                "modality": "voice",
                "emotions": emotions,
                "dominant_emotion": dominant_emotion[0],
                "confidence": dominant_emotion[1],
                "process_time": process_time
            }
            
        except Exception as e:
            logger.error(f"Voice emotion detection error: {e}", exc_info=True)
            return {
                "modality": "voice",
                "error": str(e),
                "emotions": {emotion: 0.0 for emotion in self.emotion_categories},
                "dominant_emotion": "neutral",
                "confidence": 0.0
            }
            
    def _extract_audio_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract features from audio for emotion detection."""
        try:
            import librosa
            
            # Extract common audio features
            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            
            # Energy features
            rms = librosa.feature.rms(y=audio)[0]
            
            # Combine features
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                [np.mean(spectral_centroids), np.std(spectral_centroids)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
                [np.mean(rms), np.std(rms)]
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}", exc_info=True)
            return np.zeros(30)  # Return zero features as fallback
            
    def detect_face_emotion(self, image_data: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """
        Detect emotion from facial expression.
        
        Args:
            image_data: Image as numpy array, bytes, or file path
            
        Returns:
            Dictionary with facial emotion analysis
        """
        if not self.face_model_available or not self.cv2_available:
            return {"modality": "face", "error": "Face analysis unavailable"}
            
        start_time = time.time()
        
        try:
            import cv2
            
            # Process image data to the right format
            if isinstance(image_data, bytes):
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(image_data, str):
                # Load image file
                img = cv2.imread(image_data)
            else:
                # Assume numpy array
                img = image_data
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return {
                    "modality": "face",
                    "emotions": {emotion: 0.0 for emotion in self.emotion_categories},
                    "dominant_emotion": "neutral",
                    "confidence": 0.0,
                    "faces_detected": 0
                }
                
            # Focus on largest face (closest)
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to expected size
            face_roi = cv2.resize(face_roi, (48, 48))
            
            # Check if model is trained
            if hasattr(self, 'face_model_trained') and not self.face_model_trained:
                # Model not trained, return placeholder
                emotions = {
                    "neutral": 0.6,
                    "happy": 0.2,
                    "sad": 0.05,
                    "angry": 0.05,
                    "fearful": 0.03,
                    "disgusted": 0.02,
                    "surprised": 0.03,
                    "confused": 0.01,
                    "interested": 0.01
                }
            else:
                # Use model for prediction - implementation depends on model type
                if isinstance(self.face_emotion_model, cv2.dnn.Net):
                    # OpenCV DNN model
                    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (48, 48))
                    self.face_emotion_model.setInput(blob)
                    predictions = self.face_emotion_model.forward()
                    emotions = {
                        emotion: float(pred) 
                        for emotion, pred in zip(self.emotion_categories, predictions[0])
                    }
                elif hasattr(self.face_emotion_model, 'predict_proba'):
                    # Scikit-learn style model
                    features = face_roi.flatten() / 255.0
                    probabilities = self.face_emotion_model.predict_proba([features])[0]
                    emotions = {
                        emotion: float(prob) 
                        for emotion, prob in zip(self.emotion_categories, probabilities)
                    }
                else:
                    # Assume TensorFlow/Keras model
                    face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
                    predictions = self.face_emotion_model.predict(face_roi)[0]
                    emotions = {
                        emotion: float(pred) 
                        for emotion, pred in zip(self.emotion_categories, predictions)
                    }
            
            # Determine dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            process_time = time.time() - start_time
            
            return {
                "modality": "face",
                "emotions": emotions,
                "dominant_emotion": dominant_emotion[0],
                "confidence": dominant_emotion[1],
                "process_time": process_time,
                "faces_detected": len(faces),
                "face_coordinates": largest_face.tolist()
            }
            
        except Exception as e:
            logger.error(f"Face emotion detection error: {e}", exc_info=True)
            return {
                "modality": "face",
                "error": str(e),
                "emotions": {emotion: 0.0 for emotion in self.emotion_categories},
                "dominant_emotion": "neutral",
                "confidence": 0.0,
                "faces_detected": 0
            }
            
    def _format_context_emotion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format contextual emotion information."""
        emotion_name = context.get("emotion", "neutral")
        emotion_intensity = context.get("emotion_intensity", 0.5)
        
        # Create emotion distribution with the specified emotion as dominant
        emotions = {e: 0.1 for e in self.emotion_categories}
        emotions[emotion_name] = emotion_intensity
        
        # Normalize to ensure sum is 1.0
        total = sum(emotions.values())
        emotions = {e: v/total for e, v in emotions.items()}
        
        return {
            "modality": "context",
            "emotions": emotions,
            "dominant_emotion": emotion_name,
            "confidence": emotion_intensity,
            "source": context.get("emotion_source", "conversation")
        }
        
    def _combine_emotions(self, modality_results: Dict[str, Dict[str, Any]], 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combine emotions from multiple modalities.
        
        Args:
            modality_results: Results from each modality
            context: Optional contextual information
            
        Returns:
            Combined emotion assessment
        """
        if not modality_results:
            return {
                "dominant_emotion": "neutral",
                "emotions": {emotion: 0.0 for emotion in self.emotion_categories},
                "confidence": 0.0,
                "modalities_used": []
            }
            
        # Initialize combined emotion scores
        combined_emotions = {emotion: 0.0 for emotion in self.emotion_categories}
        
        # Track modalities used and their confidence
        modalities_used = []
        modality_confidences = {}
        
        # Calculate weighted emotion scores from each modality
        total_weight = 0
        for modality, result in modality_results.items():
            # Skip modalities with errors
            if "error" in result:
                continue
                
            weight = self.modality_weights.get(modality, 0.0)
            
            # Apply confidence adjustment to weight
            confidence = result.get("confidence", 0.5)
            adjusted_weight = weight * confidence
            
            # Add emotions from this modality
            for emotion, score in result.get("emotions", {}).items():
                if emotion in combined_emotions:
                    combined_emotions[emotion] += score * adjusted_weight
                    
            total_weight += adjusted_weight
            modalities_used.append(modality)
            modality_confidences[modality] = confidence
            
        # Normalize if we have valid results
        if total_weight > 0:
            combined_emotions = {e: s/total_weight for e, s in combined_emotions.items()}
            
        # Find dominant emotion
        if combined_emotions:
            dominant_emotion = max(combined_emotions.items(), key=lambda x: x[1])
        else:
            dominant_emotion = ("neutral", 0.0)
            
        # Calculate overall confidence - weighted average of modality confidences
        if modality_confidences:
            overall_confidence = sum(
                modality_confidences[m] * self.modality_weights.get(m, 0.0) 
                for m in modalities_used
            ) / sum(self.modality_weights.get(m, 0.0) for m in modalities_used)
        else:
            overall_confidence = 0.0
            
        # Apply contextual adjustments
        if context:
            combined_emotions, dominant_emotion, overall_confidence = self._apply_contextual_adjustments(
                combined_emotions, dominant_emotion, overall_confidence, context
            )
            
        return {
            "dominant_emotion": dominant_emotion[0],
            "emotions": combined_emotions,
            "confidence": overall_confidence,
            "modalities_used": modalities_used,
            "modality_confidences": modality_confidences
        }
        
    def _apply_contextual_adjustments(self, emotions: Dict[str, float], 
                                    dominant: Tuple[str, float],
                                    confidence: float,
                                    context: Dict[str, Any]) -> Tuple[Dict[str, float], Tuple[str, float], float]:
        """Apply contextual adjustments to emotions."""
        # Make copies to avoid modifying original
        adjusted_emotions = emotions.copy()
        adjusted_dominant = dominant
        adjusted_confidence = confidence
        
        # Conversation context
        conversation_type = context.get("conversation_type", "")
        if conversation_type == "formal":
            # Formal conversations tend to have more restrained emotions
            for emotion in ["angry", "happy", "surprised"]:
                if emotion in adjusted_emotions:
                    adjusted_emotions[emotion] *= 0.8
            if "neutral" in adjusted_emotions:
                adjusted_emotions["neutral"] *= 1.2
                
        elif conversation_type == "distressed":
            # Distressed conversations amplify negative emotions
            for emotion in ["fearful", "sad", "angry"]:
                if emotion in adjusted_emotions:
                    adjusted_emotions[emotion] *= 1.2
                    
        # Relationship context
        relationship = context.get("relationship", "")
        if relationship == "master":
            # Speaking to master - adjust for deference
            if "interested" in adjusted_emotions:
                adjusted_emotions["interested"] *= 1.2
                
        # Recalculate dominant emotion
        if adjusted_emotions:
            adjusted_dominant = max(adjusted_emotions.items(), key=lambda x: x[1])
            
        # Adjust confidence based on context quality
        context_quality = context.get("context_quality", 0.5)
        adjusted_confidence = confidence * 0.8 + context_quality * 0.2
        
        return adjusted_emotions, adjusted_dominant, adjusted_confidence
        
    def map_to_sebastian_emotion(self, emotion_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map detected emotions to Sebastian-appropriate emotional responses.
        
        Args:
            emotion_result: Result from emotion detection
            
        Returns:
            Sebastian-appropriate emotion assessment
        """
        # Get dominant emotion and confidence
        dominant_emotion = emotion_result.get("dominant_emotion", "neutral")
        confidence = emotion_result.get("confidence", 0.0)
        
        # Sebastian's emotional mapping - translate common emotions to his restricted range
        sebastian_emotion_map = {
            "neutral": "neutral",
            "happy": "pleased" if confidence > 0.7 else "satisfied",
            "sad": "concerned",
            "angry": "irritated" if confidence < 0.8 else "angry",
            "fearful": "concerned",
            "disgusted": "displeased",
            "surprised": "surprised" if confidence > 0.7 else "amused",
            "confused": "concerned",
            "interested": "interested"
        }
        
        # Map to Sebastian's emotion
        sebastian_emotion = sebastian_emotion_map.get(dominant_emotion, "neutral")
        
        # Calculate intensity - Sebastian always maintains composure
        intensity = min(0.7, confidence * 0.8)
        
        # Special case for extreme emotions
        if confidence > 0.85 and dominant_emotion in ["angry", "fearful"]:
            # Even in extreme cases, Sebastian remains composed
            sebastian_emotion = "protective" if dominant_emotion == "fearful" else "displeased"
            intensity = 0.8
            
        return {
            "sebastian_emotion": sebastian_emotion,
            "intensity": intensity,
            "original_emotion": dominant_emotion,
            "original_confidence": confidence
        }