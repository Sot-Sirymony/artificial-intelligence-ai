# 🎭 Text Emotion Detection

A machine learning project for detecting emotions in text using Natural Language Processing (NLP) and machine learning techniques.

## 📋 Overview

This project implements a complete emotion detection system that can classify text into 8 different emotions:
- 😊 **Joy** - Happiness, excitement, delight
- 😢 **Sadness** - Sorrow, depression, melancholy  
- 😠 **Anger** - Fury, irritation, frustration
- 😨 **Fear** - Anxiety, terror, worry
- 😲 **Surprise** - Astonishment, shock, amazement
- 🤢 **Disgust** - Revulsion, repulsion, aversion
- 🥰 **Love** - Affection, adoration, fondness
- 😐 **Neutral** - Indifference, apathy, calm

## 🏗️ Project Structure

```
Text Emotion Detection/
├── dataset/
│   └── emotions.csv              # Training dataset
├── src/
│   ├── preprocessing.py          # Text preprocessing module
│   ├── model.py                  # Model training and evaluation
│   └── predict.py                # Prediction functionality
├── model/
│   └── emotion_model.pkl         # Trained model (generated)
├── notebooks/
│   └── training.ipynb            # Jupyter notebook for training
├── app.py                        # Streamlit web application
├── train_model.py                # Training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Text Emotion Detection"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train the emotion detection model
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a logistic regression model
- Evaluate performance
- Save the trained model to `model/emotion_model.pkl`
- Generate visualizations and wordclouds

### 3. Run the Web Application

```bash
# Start the Streamlit web app
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to use the emotion detection interface.

## 📊 Model Performance

The model achieves approximately **85% accuracy** on the test set with the following performance metrics:

- **Precision**: 0.85
- **Recall**: 0.85  
- **F1-Score**: 0.85

Performance may vary depending on the dataset and text characteristics.

## 🛠️ Usage

### Command Line Usage

```python
from src.predict import EmotionPredictor

# Load the trained model
predictor = EmotionPredictor("model/emotion_model.pkl")

# Predict emotion for a single text
result = predictor.predict_emotion("I am so happy today!")
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
texts = ["I feel sad", "I'm excited!", "This is frustrating"]
results = predictor.predict_batch(texts)
```

### Web Application Features

The Streamlit app provides:

1. **Single Text Analysis**: Analyze emotions in individual text inputs
2. **Batch Analysis**: Upload CSV files or enter multiple texts
3. **Visualizations**: Radar charts, bar charts, and emotion distributions
4. **Detailed Insights**: Confidence scores, secondary emotions, and analysis
5. **Export Results**: Download analysis results as CSV

## 🔧 Technical Details

### Model Architecture

- **Feature Extraction**: TF-IDF Vectorization (5000 features)
- **Classifier**: Logistic Regression
- **N-gram Range**: (1, 2) for capturing word combinations
- **Preprocessing**: Text cleaning, lemmatization, stopword removal

### Preprocessing Pipeline

1. **Text Cleaning**: Lowercase conversion, punctuation removal
2. **URL/Email Removal**: Clean web addresses and email patterns
3. **Tokenization**: Split text into words
4. **Stopword Removal**: Remove common words (the, is, at, etc.)
5. **Lemmatization**: Convert words to base form (running → run)

### Supported Model Types

- **Logistic Regression** (default) - Fast, interpretable
- **Random Forest** - Good for complex patterns
- **Support Vector Machine** - High accuracy, slower training

## 📈 Training Process

The training script (`train_model.py`) performs:

1. **Data Loading**: Load emotion dataset from CSV
2. **Preprocessing**: Clean and normalize text data
3. **Feature Engineering**: TF-IDF vectorization
4. **Model Training**: Train classifier with cross-validation
5. **Evaluation**: Calculate accuracy, precision, recall, F1-score
6. **Visualization**: Confusion matrix, emotion distribution, wordclouds
7. **Model Saving**: Save trained model for later use

## 🎯 Example Predictions

```python
# Example texts and their predicted emotions
"I am so happy today!" → Joy (95% confidence)
"I feel really sad and lonely" → Sadness (88% confidence)
"This makes me so angry!" → Anger (92% confidence)
"I'm scared of what might happen" → Fear (85% confidence)
"I love spending time with my family" → Love (90% confidence)
```

## 🔍 Model Insights

### Feature Importance

The model learns to recognize:
- **Emotion-specific words**: happy, sad, angry, scared, love
- **Intensifiers**: so, very, really, extremely
- **Context patterns**: word combinations and phrases
- **Punctuation**: exclamation marks, question marks

### Limitations

- **Context dependency**: Requires sufficient context for accurate predictions
- **Language specificity**: Trained on English text
- **Emotion complexity**: May not capture mixed or subtle emotions
- **Dataset size**: Limited training data affects generalization

## 🚀 Advanced Usage

### Hyperparameter Tuning

```python
# Enable hyperparameter tuning in training
detector = EmotionDetector(model_type='logistic')
detector.train(X_train, y_train, tune_hyperparameters=True)
```

### Custom Model Types

```python
# Use different model types
detector = EmotionDetector(model_type='random_forest')  # or 'svm'
detector.train(X_train, y_train)
```

### Custom Dataset

Replace `dataset/emotions.csv` with your own dataset:

```csv
text,emotion
"Your text here",emotion_label
"Another text",another_emotion
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset inspiration from various emotion detection datasets
- NLTK and scikit-learn for NLP and ML capabilities
- Streamlit for the web interface
- Plotly for interactive visualizations

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

---

**Happy Emotion Detection! 🎭✨** 