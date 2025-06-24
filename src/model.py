import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class EmotionDetector:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.model = None
        self.pipeline = None
        self.emotions = None
        
    def create_model(self):
        """Create the specified model"""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_estimators=100)
        elif self.model_type == 'svm':
            self.model = SVC(random_state=42, probability=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return self.model
    
    def create_pipeline(self):
        """Create a pipeline with vectorizer and model"""
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.create_model())
        ])
        return self.pipeline
    
    def train(self, X_train, y_train, tune_hyperparameters=False):
        """Train the model"""
        if self.pipeline is None:
            self.create_pipeline()
            
        if tune_hyperparameters:
            self._tune_hyperparameters(X_train, y_train)
        else:
            self.pipeline.fit(X_train, y_train)
            
        return self.pipeline
    
    def _tune_hyperparameters(self, X_train, y_train):
        """Tune hyperparameters using GridSearchCV"""
        if self.model_type == 'logistic':
            param_grid = {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2']
            }
        elif self.model_type == 'random_forest':
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None]
            }
        elif self.model_type == 'svm':
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['rbf', 'linear']
            }
            
        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        self.pipeline = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    def predict(self, X):
        """Make predictions"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.pipeline.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def plot_confusion_matrix(self, confusion_matrix, emotions, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=emotions,
            yticklabels=emotions
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_emotion_distribution(self, y, save_path=None):
        """Plot emotion distribution"""
        emotion_counts = pd.Series(y).value_counts()
        
        plt.figure(figsize=(12, 6))
        emotion_counts.plot(kind='bar')
        plt.title('Emotion Distribution in Dataset')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def create_wordcloud(self, texts, emotion, save_path=None):
        """Create wordcloud for specific emotion"""
        emotion_texts = ' '.join(texts)
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100
        ).generate(emotion_texts)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {emotion} emotion')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.pipeline is None:
            raise ValueError("No trained model to save")
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.pipeline = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.pipeline 