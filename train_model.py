#!/usr/bin/env python3
"""
Emotion Detection Model Training Script
=======================================

This script trains an emotion detection model using the provided dataset.
It includes data preprocessing, model training, evaluation, and visualization.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import TextPreprocessor
from src.model import EmotionDetector

def load_dataset(filepath):
    """Load the emotion dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully: {len(df)} samples")
        print(f"📊 Dataset columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n🧹 Preprocessing data...")
    
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataset(df, text_column='text')
    
    # Remove empty texts after preprocessing
    df_clean = df_clean[df_clean['text_clean'].str.strip() != '']
    
    print(f"✅ Preprocessing completed: {len(df_clean)} samples remaining")
    print(f"📝 Sample cleaned text: '{df_clean['text_clean'].iloc[0]}'")
    
    return df_clean

def analyze_dataset(df):
    """Analyze the dataset"""
    print("\n📊 Dataset Analysis:")
    print("=" * 50)
    
    # Emotion distribution
    emotion_counts = df['emotion'].value_counts()
    print(f"🎭 Emotion distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {emotion:12}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Text length statistics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\n📏 Text statistics:")
    print(f"   Average text length: {df['text_length'].mean():.1f} characters")
    print(f"   Average word count: {df['word_count'].mean():.1f} words")
    print(f"   Min text length: {df['text_length'].min()} characters")
    print(f"   Max text length: {df['text_length'].max()} characters")
    
    return emotion_counts

def train_model(X_train, y_train, model_type='logistic', tune_hyperparameters=False):
    """Train the emotion detection model"""
    print(f"\n🤖 Training {model_type} model...")
    
    detector = EmotionDetector(model_type=model_type)
    
    if tune_hyperparameters:
        print("🔧 Tuning hyperparameters...")
        detector.train(X_train, y_train, tune_hyperparameters=True)
    else:
        detector.train(X_train, y_train, tune_hyperparameters=False)
    
    print("✅ Model training completed!")
    return detector

def evaluate_model(detector, X_test, y_test, emotions):
    """Evaluate the trained model"""
    print("\n📈 Evaluating model...")
    
    results = detector.evaluate(X_test, y_test)
    
    print(f"🎯 Accuracy: {results['accuracy']:.3f}")
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    print("\n📊 Plotting confusion matrix...")
    detector.plot_confusion_matrix(results['confusion_matrix'], emotions)
    
    # Plot emotion distribution
    print("\n📊 Plotting emotion distribution...")
    detector.plot_emotion_distribution(y_test)
    
    return results

def create_wordclouds(detector, df, emotions):
    """Create wordclouds for each emotion"""
    print("\n☁️ Creating wordclouds...")
    
    for emotion in emotions:
        emotion_texts = df[df['emotion'] == emotion]['text_clean'].tolist()
        if emotion_texts:
            detector.create_wordcloud(emotion_texts, emotion)

def save_model(detector, model_path):
    """Save the trained model"""
    print(f"\n💾 Saving model to {model_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    detector.save_model(model_path)
    print("✅ Model saved successfully!")

def main():
    """Main training function"""
    print("🎭 Emotion Detection Model Training")
    print("=" * 50)
    
    # Configuration
    dataset_path = "dataset/emotions.csv"
    model_path = "model/emotion_model.pkl"
    model_type = "logistic"  # Options: 'logistic', 'random_forest', 'svm'
    tune_hyperparameters = False  # Set to True for hyperparameter tuning
    test_size = 0.2
    random_state = 42
    
    # Load dataset
    df = load_dataset(dataset_path)
    if df is None:
        return
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Analyze dataset
    emotion_counts = analyze_dataset(df_clean)
    emotions = emotion_counts.index.tolist()
    
    # Split data
    print(f"\n✂️ Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean['text_clean'], 
        df_clean['emotion'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df_clean['emotion']
    )
    
    print(f"📚 Training set: {len(X_train)} samples")
    print(f"🧪 Test set: {len(X_test)} samples")
    
    # Train model
    detector = train_model(X_train, y_train, model_type, tune_hyperparameters)
    
    # Evaluate model
    results = evaluate_model(detector, X_test, y_test, emotions)
    
    # Create wordclouds
    create_wordclouds(detector, df_clean, emotions)
    
    # Save model
    save_model(detector, model_path)
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Model saved to: {model_path}")
    print(f"🎯 Final accuracy: {results['accuracy']:.3f}")
    
    # Test with sample predictions
    print("\n🧪 Testing with sample predictions:")
    test_texts = [
        "I am so happy today!",
        "I feel really sad and lonely",
        "This makes me so angry!",
        "I'm scared of what might happen",
        "I love spending time with my family"
    ]
    
    for text in test_texts:
        prediction = detector.predict([text])[0]
        print(f"   '{text}' → {prediction}")

if __name__ == "__main__":
    main() 