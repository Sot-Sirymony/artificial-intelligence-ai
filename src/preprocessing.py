import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Clean and preprocess text for emotion detection
        """
        if pd.isna(text) or text == '':
            return ''
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_dataset(self, df, text_column='text'):
        """
        Preprocess entire dataset
        """
        df_clean = df.copy()
        df_clean['text_clean'] = df_clean[text_column].apply(self.clean_text)
        return df_clean
    
    def get_emotion_mapping(self):
        """
        Return emotion label mapping
        """
        return {
            'joy': 'joy',
            'happiness': 'joy',
            'happy': 'joy',
            'excited': 'joy',
            'sad': 'sadness',
            'sadness': 'sadness',
            'depressed': 'sadness',
            'angry': 'anger',
            'anger': 'anger',
            'furious': 'anger',
            'fear': 'fear',
            'scared': 'fear',
            'terrified': 'fear',
            'surprise': 'surprise',
            'surprised': 'surprise',
            'shocked': 'surprise',
            'disgust': 'disgust',
            'disgusted': 'disgust',
            'love': 'love',
            'loving': 'love',
            'neutral': 'neutral'
        } 