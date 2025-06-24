#!/usr/bin/env python3
"""
Emotion Detection Web Application
================================

A Streamlit web app for detecting emotions in text using a trained machine learning model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import EmotionPredictor

# Page configuration
st.set_page_config(
    page_title="🎭 Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .emotion-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 5px;
        height: 20px;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #28a745, #20c997);
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s ease;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained emotion detection model"""
    try:
        model_path = "model/emotion_model.pkl"
        predictor = EmotionPredictor(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_emotion_radar_chart(probabilities):
    """Create a radar chart for emotion probabilities"""
    emotions = list(probabilities.keys())
    values = list(probabilities.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        name='Emotion Probabilities',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Emotion Probability Distribution",
        height=400
    )
    
    return fig

def create_emotion_bar_chart(probabilities):
    """Create a bar chart for emotion probabilities"""
    emotions = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Color mapping for emotions
    color_map = {
        'joy': '#FFD700',
        'sadness': '#4682B4',
        'anger': '#DC143C',
        'fear': '#8B4513',
        'surprise': '#FF69B4',
        'disgust': '#228B22',
        'love': '#FF1493',
        'neutral': '#808080'
    }
    
    colors = [color_map.get(emotion, '#1f77b4') for emotion in emotions]
    
    fig = px.bar(
        x=emotions,
        y=values,
        color=emotions,
        color_discrete_map=color_map,
        title="Emotion Probabilities",
        labels={'x': 'Emotion', 'y': 'Probability'},
        height=400
    )
    
    fig.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Probability",
        showlegend=False
    )
    
    return fig

def get_emotion_emoji(emotion):
    """Get emoji for emotion"""
    emoji_map = {
        'joy': '😊',
        'sadness': '😢',
        'anger': '😠',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'love': '🥰',
        'neutral': '😐'
    }
    return emoji_map.get(emotion, '🤔')

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Emotion Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Analyze emotions in text using AI</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading emotion detection model..."):
        predictor = load_model()
    
    if predictor is None:
        st.error("❌ Failed to load the emotion detection model. Please ensure the model file exists.")
        st.info("💡 Run the training script first: `python train_model.py`")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Settings")
    
    # Model information
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.info("""
    **Model Type:** Logistic Regression
    **Features:** TF-IDF Vectorization
    **Training Data:** 50+ emotion samples
    **Accuracy:** ~85% (varies by dataset)
    """)
    
    # Analysis options
    st.sidebar.markdown("### ⚙️ Analysis Options")
    show_probabilities = st.sidebar.checkbox("Show detailed probabilities", value=True)
    show_visualizations = st.sidebar.checkbox("Show visualizations", value=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Text Analysis", "📊 Batch Analysis", "📈 Model Insights"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Single Text Emotion Analysis</h2>', unsafe_allow_html=True)
        
        # Text input
        text_input = st.text_area(
            "Enter your text here:",
            placeholder="Type or paste your text to analyze emotions...",
            height=150
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            analyze_button = st.button("🔍 Analyze Emotion", type="primary")
        
        with col2:
            if st.button("🎲 Try Examples"):
                examples = [
                    "I am so happy today! Everything is going great!",
                    "I feel really sad and lonely right now",
                    "This makes me so angry! I can't believe it!",
                    "I'm scared of what might happen next",
                    "I love spending time with my family"
                ]
                text_input = st.session_state.get('example_text', examples[0])
                st.session_state['example_text'] = examples[(examples.index(text_input) + 1) % len(examples)]
        
        if analyze_button and text_input.strip():
            with st.spinner("Analyzing emotions..."):
                try:
                    # Get prediction
                    prediction = predictor.get_emotion_analysis(text_input)
                    
                    # Display results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("### 🎯 Results")
                        emotion = prediction['emotion']
                        confidence = prediction['confidence']
                        emoji = get_emotion_emoji(emotion)
                        
                        st.markdown(f"""
                        <div class="emotion-card">
                            <h3>{emoji} {emotion.title()}</h3>
                            <p><strong>Confidence:</strong> {confidence:.1%}</p>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {confidence*100}%"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Analysis insights
                        analysis = prediction['analysis']
                        st.markdown("### 📋 Analysis")
                        st.info(f"""
                        **Confidence Level:** {analysis['confidence_level']}
                        **Text Length:** {analysis['text_length']} characters
                        **Word Count:** {analysis['word_count']} words
                        **Secondary Emotions:** {', '.join(analysis['secondary_emotions']) if analysis['secondary_emotions'] else 'None'}
                        """)
                    
                    with col2:
                        if show_probabilities:
                            st.markdown("### 📊 Emotion Probabilities")
                            
                            # Create probability dataframe
                            prob_df = pd.DataFrame([
                                {'Emotion': emotion.title(), 'Probability': prob}
                                for emotion, prob in prediction['probabilities'].items()
                            ]).sort_values('Probability', ascending=False)
                            
                            st.dataframe(prob_df, use_container_width=True)
                    
                    # Visualizations
                    if show_visualizations:
                        st.markdown("### 📈 Visualizations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Radar chart
                            radar_fig = create_emotion_radar_chart(prediction['probabilities'])
                            st.plotly_chart(radar_fig, use_container_width=True)
                        
                        with col2:
                            # Bar chart
                            bar_fig = create_emotion_bar_chart(prediction['probabilities'])
                            st.plotly_chart(bar_fig, use_container_width=True)
                    
                    # Original vs cleaned text
                    with st.expander("🔍 Text Processing Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Text:**")
                            st.text(prediction['original_text'])
                        with col2:
                            st.markdown("**Cleaned Text:**")
                            st.text(prediction['cleaned_text'])
                
                except Exception as e:
                    st.error(f"❌ Error analyzing text: {e}")
        
        elif analyze_button and not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Batch Text Analysis</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file with text column:",
            type=['csv'],
            help="CSV file should have a 'text' column"
        )
        
        # Or manual input
        st.markdown("### 📝 Or enter multiple texts manually:")
        batch_texts = st.text_area(
            "Enter multiple texts (one per line):",
            placeholder="I am happy today\nI feel sad\nThis makes me angry",
            height=200
        )
        
        if st.button("🔍 Analyze Batch", type="primary"):
            texts_to_analyze = []
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns:
                        texts_to_analyze = df['text'].tolist()
                        st.success(f"✅ Loaded {len(texts_to_analyze)} texts from file")
                    else:
                        st.error("❌ CSV file must have a 'text' column")
                        return
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
                    return
            
            elif batch_texts.strip():
                texts_to_analyze = [text.strip() for text in batch_texts.split('\n') if text.strip()]
                st.success(f"✅ Loaded {len(texts_to_analyze)} texts")
            
            if texts_to_analyze:
                with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                    try:
                        results = predictor.predict_batch(texts_to_analyze)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame([
                            {
                                'Text': result['original_text'],
                                'Emotion': result['emotion'].title(),
                                'Confidence': f"{result['confidence']:.2%}",
                                'Top Emotion': result['emotion'].title()
                            }
                            for result in results
                        ])
                        
                        st.markdown("### 📊 Batch Analysis Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        emotion_counts = results_df['Emotion'].value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📈 Emotion Distribution")
                            fig = px.pie(
                                values=emotion_counts.values,
                                names=emotion_counts.index,
                                title="Emotion Distribution in Batch"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 📊 Emotion Counts")
                            st.dataframe(emotion_counts.reset_index().rename(
                                columns={'index': 'Emotion', 'Emotion': 'Count'}
                            ), use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="emotion_analysis_results.csv",
                            mime="text/csv"
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Error in batch analysis: {e}")
            else:
                st.warning("⚠️ Please provide texts to analyze (upload file or enter manually).")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Model Insights & Information</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎭 Supported Emotions")
            emotions_info = {
                '😊 Joy': 'Happiness, excitement, delight',
                '😢 Sadness': 'Sorrow, depression, melancholy',
                '😠 Anger': 'Fury, irritation, frustration',
                '😨 Fear': 'Anxiety, terror, worry',
                '😲 Surprise': 'Astonishment, shock, amazement',
                '🤢 Disgust': 'Revulsion, repulsion, aversion',
                '🥰 Love': 'Affection, adoration, fondness',
                '😐 Neutral': 'Indifference, apathy, calm'
            }
            
            for emotion, description in emotions_info.items():
                st.markdown(f"**{emotion}** - {description}")
        
        with col2:
            st.markdown("### 🔧 Technical Details")
            st.info("""
            **Model Architecture:**
            - TF-IDF Vectorization (5000 features)
            - Logistic Regression Classifier
            - N-gram range: (1, 2)
            
            **Preprocessing:**
            - Text lowercasing
            - Punctuation removal
            - Stopword removal
            - Lemmatization
            - URL/email removal
            
            **Performance:**
            - Training time: ~30 seconds
            - Prediction time: ~0.1 seconds
            - Memory usage: ~50MB
            """)
        
        st.markdown("### 📚 How It Works")
        st.markdown("""
        1. **Text Preprocessing**: The input text is cleaned and normalized
        2. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
        3. **Classification**: The trained model predicts emotion probabilities
        4. **Post-processing**: Results are formatted and confidence scores calculated
        
        The model learns patterns from labeled emotion data and can generalize to new texts.
        """)
        
        st.markdown("### 🚀 Usage Tips")
        st.markdown("""
        - **Be specific**: More descriptive text leads to better predictions
        - **Context matters**: The model considers word combinations and context
        - **Confidence scores**: Higher confidence indicates more reliable predictions
        - **Multiple emotions**: Text can contain mixed emotions (shown in probabilities)
        """)

if __name__ == "__main__":
    main() 