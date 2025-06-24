import pandas as pd
import numpy as np
from .preprocessing import TextPreprocessor
from .model import EmotionDetector
import joblib

class EmotionPredictor:
    def __init__(self, model_path=None):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.emotions = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_emotion(self, text):
        """
        Predict emotion for a single text input
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Preprocess the text
        cleaned_text = self.preprocessor.clean_text(text)
        
        if not cleaned_text.strip():
            return {
                'emotion': 'neutral',
                'confidence': 1.0,
                'probabilities': {'neutral': 1.0},
                'original_text': text,
                'cleaned_text': cleaned_text
            }
        
        # Make prediction
        prediction = self.model.predict([cleaned_text])[0]
        probabilities = self.model.predict_proba([cleaned_text])[0]
        
        # Get emotion labels
        if hasattr(self.model, 'classes_'):
            emotion_labels = self.model.classes_
        else:
            # Default emotions if not available
            emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'love', 'neutral']
        
        # Create probability dictionary
        prob_dict = dict(zip(emotion_labels, probabilities))
        
        # Get confidence (max probability)
        confidence = max(probabilities)
        
        return {
            'emotion': prediction,
            'confidence': confidence,
            'probabilities': prob_dict,
            'original_text': text,
            'cleaned_text': cleaned_text
        }
    
    def predict_batch(self, texts):
        """
        Predict emotions for multiple texts
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        results = []
        
        for text in texts:
            result = self.predict_emotion(text)
            results.append(result)
        
        return results
    
    def get_emotion_analysis(self, text):
        """
        Get detailed emotion analysis with insights
        """
        prediction = self.predict_emotion(text)
        
        # Add analysis insights
        analysis = {
            **prediction,
            'analysis': {
                'primary_emotion': prediction['emotion'],
                'confidence_level': self._get_confidence_level(prediction['confidence']),
                'secondary_emotions': self._get_secondary_emotions(prediction['probabilities']),
                'text_length': len(text),
                'word_count': len(text.split()),
                'has_emotion_words': self._check_emotion_words(text)
            }
        }
        
        return analysis
    
    def _get_confidence_level(self, confidence):
        """Convert confidence score to level"""
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _get_secondary_emotions(self, probabilities, threshold=0.1):
        """Get secondary emotions with probability above threshold"""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        secondary = [emotion for emotion, prob in sorted_probs[1:] if prob > threshold]
        return secondary[:3]  # Return top 3 secondary emotions
    
    def _check_emotion_words(self, text):
        """Check if text contains emotion-related words"""
        emotion_words = {
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'gloomy'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished'],
            'disgust': ['disgusted', 'revolted', 'sickened'],
            'love': ['love', 'adore', 'cherish', 'affectionate']
        }
        
        text_lower = text.lower()
        found_emotions = []
        
        for emotion, words in emotion_words.items():
            if any(word in text_lower for word in words):
                found_emotions.append(emotion)
        
        return found_emotions
    
    def format_prediction_output(self, prediction, format_type='simple'):
        """
        Format prediction output in different styles
        """
        if format_type == 'simple':
            return f"Emotion: {prediction['emotion'].title()} (Confidence: {prediction['confidence']:.2f})"
        
        elif format_type == 'detailed':
            output = f"""
📝 Text Analysis Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📄 Original Text: "{prediction['original_text']}"
🧹 Cleaned Text: "{prediction['cleaned_text']}"

😊 Detected Emotion: {prediction['emotion'].title()}
🎯 Confidence: {prediction['confidence']:.2%}

📊 Emotion Probabilities:
"""
            for emotion, prob in sorted(prediction['probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
                bar_length = int(prob * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                output += f"   {emotion.title():12} {bar} {prob:.2%}\n"
            
            return output
        
        elif format_type == 'json':
            return prediction
        
        else:
            return str(prediction)

def predict_emotion_simple(text, model_path='model/emotion_model.pkl'):
    """
    Simple function to predict emotion for a single text
    """
    try:
        predictor = EmotionPredictor(model_path)
        result = predictor.predict_emotion(text)
        return result['emotion'], result['confidence']
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return 'neutral', 0.0 