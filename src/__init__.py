"""
Text Emotion Detection Package
==============================

A machine learning package for detecting emotions in text using NLP techniques.

Modules:
- preprocessing: Text preprocessing and cleaning
- model: Model training and evaluation
- predict: Prediction functionality
"""

from .preprocessing import TextPreprocessor
from .model import EmotionDetector
from .predict import EmotionPredictor, predict_emotion_simple

__version__ = "1.0.0"
__author__ = "Emotion Detection Team"

__all__ = [
    'TextPreprocessor',
    'EmotionDetector', 
    'EmotionPredictor',
    'predict_emotion_simple'
] 