#!/usr/bin/env python3
"""
Test Script for Emotion Detection Model
=======================================

This script tests the trained emotion detection model with sample texts.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import EmotionPredictor

def test_model():
    """Test the emotion detection model"""
    print("🧪 Testing Emotion Detection Model")
    print("=" * 50)
    
    # Test texts with expected emotions
    test_cases = [
        ("I am so happy today! Everything is going great!", "joy"),
        ("I feel really sad and lonely right now", "sadness"),
        ("This makes me so angry! I can't believe it!", "anger"),
        ("I'm scared of what might happen next", "fear"),
        ("I'm so surprised by this unexpected news!", "surprise"),
        ("This is disgusting, I can't stand it", "disgust"),
        ("I love spending time with my family", "love"),
        ("I'm feeling neutral about this situation", "neutral"),
        ("What a wonderful day! I'm excited!", "joy"),
        ("I'm feeling down and depressed today", "sadness")
    ]
    
    try:
        # Load the model
        print("📂 Loading model...")
        predictor = EmotionPredictor("model/emotion_model.pkl")
        print("✅ Model loaded successfully!")
        
        # Test predictions
        print("\n🔍 Testing predictions:")
        print("-" * 50)
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for i, (text, expected_emotion) in enumerate(test_cases, 1):
            try:
                # Get prediction
                result = predictor.predict_emotion(text)
                predicted_emotion = result['emotion']
                confidence = result['confidence']
                
                # Check if prediction is correct
                is_correct = predicted_emotion == expected_emotion
                if is_correct:
                    correct_predictions += 1
                
                # Display result
                status = "✅" if is_correct else "❌"
                print(f"{i:2d}. {status} '{text[:50]}{'...' if len(text) > 50 else ''}'")
                print(f"    Expected: {expected_emotion:10} | Predicted: {predicted_emotion:10} | Confidence: {confidence:.1%}")
                print()
                
            except Exception as e:
                print(f"{i:2d}. ❌ Error predicting: {e}")
                print()
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions
        print(f"📊 Test Results:")
        print(f"   Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.1%}")
        
        if accuracy >= 0.7:
            print("🎉 Model is working well!")
        elif accuracy >= 0.5:
            print("⚠️ Model performance is moderate. Consider retraining with more data.")
        else:
            print("❌ Model performance is poor. Please check the training data and retrain.")
        
        # Test detailed analysis
        print("\n🔍 Testing detailed analysis:")
        print("-" * 50)
        
        sample_text = "I am absolutely thrilled and excited about this amazing opportunity!"
        analysis = predictor.get_emotion_analysis(sample_text)
        
        print(f"📝 Sample text: '{sample_text}'")
        print(f"😊 Primary emotion: {analysis['emotion'].title()}")
        print(f"🎯 Confidence: {analysis['confidence']:.1%}")
        print(f"📊 Confidence level: {analysis['analysis']['confidence_level']}")
        print(f"📏 Text length: {analysis['analysis']['text_length']} characters")
        print(f"📝 Word count: {analysis['analysis']['word_count']} words")
        print(f"🎭 Secondary emotions: {', '.join(analysis['analysis']['secondary_emotions']) if analysis['analysis']['secondary_emotions'] else 'None'}")
        
        # Test batch prediction
        print("\n📊 Testing batch prediction:")
        print("-" * 50)
        
        batch_texts = [
            "I'm feeling great today!",
            "This is so frustrating",
            "I'm terrified of spiders"
        ]
        
        batch_results = predictor.predict_batch(batch_texts)
        
        for i, (text, result) in enumerate(zip(batch_texts, batch_results), 1):
            print(f"{i}. '{text}' → {result['emotion'].title()} ({result['confidence']:.1%})")
        
        print("\n🎉 All tests completed successfully!")
        
    except FileNotFoundError:
        print("❌ Model file not found!")
        print("💡 Please run the training script first: python train_model.py")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_model() 