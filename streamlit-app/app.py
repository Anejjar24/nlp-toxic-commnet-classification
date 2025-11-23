# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

# Add utils to path
sys.path.append('utils')

from model_loader import ModelLoader

# Page configuration
st.set_page_config(
    page_title="Toxic Comments Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #262730;
        margin-bottom: 1rem;
    }
    .model-section {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
    .cnn-section {
        border-left: 5px solid #007bff;
    }
    .bert-section {
        border-left: 5px solid #28a745;
    }
    .toxicity-bar {
        height: 25px;
        border-radius: 12px;
        margin: 8px 0;
        background: #f0f0f0;
        overflow: hidden;
    }
    .toxicity-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 0.5s ease;
    }
    .model-comparison {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
            
    .binary-badge {
        background: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .agreement-good {
        color: #4CAF50;
        font-weight: bold;
    }
    .agreement-warning {
        color: #FFC107;
        font-weight: bold;
    }
    .pastel-purple-highlight {
        background-color: #e6e6ff;
        padding: 4px 8px;
        border-radius: 5px;
        font-weight: bold;
    }
    .detected-classes {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 8px;
        margin-top: 5px;
        font-size: 0.85rem;
    }
    .toxicity-badge {
        display: inline-block;
        background: #ff6b6b;
        color: white;
        padding: 2px 6px;
        border-radius: 10px;
        margin: 1px;
        font-size: 0.75rem;
        font-weight: bold;
    }
            
    .toxicity-badge1 {
    display: inline-block;
    background: #ff6b6b;
    color: white;
    padding: 6px 12px;         /* increased padding */
    border-radius: 14px;       /* slightly rounder */
    margin: 3px;
    font-size: 1rem;           /* bigger text */
    font-weight: 600;          /* bolder */
    letter-spacing: 0.3px;     /* cleaner text spacing */
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);  /* subtle shadow */
    transition: transform 0.2s ease;
}
.toxicity-badge1:hover {
    transform: scale(1.05);
}


</style>
""", unsafe_allow_html=True)

class ToxicCommentApp:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.cnn_loaded = False
        self.bert_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load both CNN and BERT models"""
        with st.spinner("🔄 Loading AI models... This may take a few moments."):
            col1, col2 = st.columns(2)
            
            with col1:
                cnn_status = st.empty()
                cnn_status.info(" Loading CNN Model...")
                self.cnn_loaded = self.model_loader.load_cnn_model()
                if self.cnn_loaded:
                    cnn_status.success("✅ CNN Model Loaded Successfully!")
                else:
                    cnn_status.error("❌ CNN Model Failed to Load")
            
            with col2:
                bert_status = st.empty()
                bert_status.info("🤖 Loading BERT Model...")
                self.bert_loaded = self.model_loader.load_bert_model()
                if self.bert_loaded:
                    bert_status.success("✅ BERT Model Loaded Successfully!")
                else:
                    bert_status.warning("⚠️ BERT Model Not Available")
            
            # Display overall status
            if self.cnn_loaded and self.bert_loaded:
                st.balloons()
            elif self.cnn_loaded:
                st.warning("⚠️ Only CNN model loaded. BERT model will not be available.")
            elif self.bert_loaded:
                st.warning("⚠️ Only BERT model loaded. CNN model will not be available.")
            else:
                st.error("❌ No models could be loaded. Please check the setup instructions.")

def main():
    # Initialize app
    app = ToxicCommentApp()
    
    # Header section
    st.markdown("""
 
    """, unsafe_allow_html=True)
    
    if not app.cnn_loaded and not app.bert_loaded:
        st.error("""
        **Setup Instructions:**
        - Ensure `cnn_model.h5` is in `models/` folder
        - Ensure `cnn_tokenizer.pkl` is in `models/tokenizers/` folder
        - For BERT: Place model files in `models/bert_model/` folder
        """)
        return
    
    # Text input section - now takes full width
    
    comment_text = st.text_area(
        "",
        height=200,
        placeholder="Type or paste your text here to analyze for toxic content...",
        help="Both CNN and BERT models will analyze this text for toxicity"
    )
    
    # Analysis button
    analyze_clicked = st.button(
        "Analyze", 
        type="primary", 
        use_container_width=True,
        disabled=not comment_text.strip()
    )
    
    # Perform analysis when button is clicked
    if analyze_clicked and comment_text.strip():
        perform_analysis(app, comment_text)

def perform_analysis(app, text):
    """Perform analysis and display results"""
    st.markdown("---")
    st.markdown('<h2 class="sub-header"> Analysis Results</h2>', unsafe_allow_html=True)
    
    # Create progress bar for analysis
    #progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize results
    cnn_results = None
    bert_results = None
    
    # Step 1: CNN Prediction
    if app.cnn_loaded:
        status_text.text(" Running CNN analysis...")
        #progress_bar.progress(30)
        
        try:
            cnn_results = app.model_loader.predict_cnn_toxicity(text)
        except Exception as e:
            st.error(f"❌ CNN analysis failed: {e}")
    
    # Step 2: BERT Prediction
    if app.bert_loaded:
        status_text.text("🤖 Running BERT analysis...")
        #progress_bar.progress(70)
        
        try:
            bert_results = app.model_loader.predict_bert_toxicity(text)
        except Exception as e:
            st.error(f"❌ BERT analysis failed: {e}")
    
    # Step 3: Display results
    status_text.text("🎯 Generating results...")
    #.progress(100)
    
    # Display results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        if app.cnn_loaded and cnn_results:
            display_cnn_results(cnn_results, text)
        elif app.cnn_loaded:
            st.error("❌ CNN analysis failed")
    
    with col2:
        if app.bert_loaded and bert_results:
            display_bert_results(bert_results, text)
        elif app.bert_loaded:
            st.error("❌ BERT analysis failed")
    
    
    
    # Clear progress
   # progress_bar.empty()
    status_text.empty()
    


def display_cnn_results(predictions, text):
    """Display CNN prediction results avec visualisations parallèles"""
    st.markdown('<div class="model-section cnn-section">', unsafe_allow_html=True)
    # --- Detected tags line ---
    detected_classes = [
        label.replace('_', ' ').title()
        for label, score in predictions.items()
        if label != 'overall_toxic' and score > 0.5
    ]
    if detected_classes:
        st.markdown(
            " ".join([f"<span class='toxicity-badge1'>{cls}</span>" for cls in detected_classes]),
            unsafe_allow_html=True
        )
    st.markdown("### <span class='pastel-purple-highlight'>CNN Analysis</span>", unsafe_allow_html=True)
    
    # Overall toxicity
    overall_toxic = predictions['overall_toxic']
    is_toxic = overall_toxic > 0.5
    
    # Get detected classes above threshold
    detected_classes = []
    for label, score in predictions.items():
        if label != 'overall_toxic' and score > 0.5:
            formatted_label = label.replace('_', ' ').title()
            detected_classes.append(formatted_label)
    
    # Main results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Toxicity", 
            f"{overall_toxic:.1%}",
            delta="TOXIC" if is_toxic else "CLEAN",
            delta_color="inverse" if is_toxic else "normal"
        )
    
    with col2:
        if is_toxic:
            st.metric("Classification", "🚨 TOXIC")
            if detected_classes:
                st.markdown(f"""
                
                """, unsafe_allow_html=True)
        else:
            st.metric("Classification", "✅ CLEAN")
    
    with col3:
        toxic_categories = len(detected_classes)
        st.metric("Toxic Categories", f"{toxic_categories}/6")

    # Probability Distribution
    st.markdown("#### <span class='pastel-purple-highlight'>📈 Probability Distribution</span>", unsafe_allow_html=True)
    individual_scores = {k: v for k, v in predictions.items() if k != 'overall_toxic'}
    
    for tox_type, score in individual_scores.items():
        display_toxicity_bar_with_binary(tox_type, score, "")

    # Radar Chart
    st.markdown("#### <span class='pastel-purple-highlight'>🎯 Toxicity Radar</span>", unsafe_allow_html=True)
    categories = [k.replace('_', ' ').title() for k in individual_scores.keys()]
    values = list(individual_scores.values())
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='CNN Scores',
        line=dict(color='#007bff', width=2),
        fillcolor='rgba(0, 123, 255, 0.3)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0', '0.25', '0.5', '0.75', '1']
            )),
        showlegend=False,
        height=400,
        title="CNN Toxicity Radar Chart"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Toxicity Scores Bar Chart
    st.markdown("#### <span class='pastel-purple-highlight'>Toxicity Scores Bar Chart</span>", unsafe_allow_html=True)
    df_cnn = pd.DataFrame({
        'Category': [k.replace('_', ' ').title() for k in individual_scores.keys()],
        'Score': list(individual_scores.values())
    })
    
    fig_bar = px.bar(
        df_cnn, 
        y='Category', 
        x='Score',
        orientation='h',
        title="CNN Toxicity Scores",
        color='Score',
        color_continuous_scale=['#4CAF50', '#FFC107', '#ff4b4b'],
        range_color=[0, 1]
    )
    fig_bar.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

def display_bert_results(predictions, text):
    """Display BERT prediction results avec visualisations parallèles"""
    st.markdown('<div class="model-section bert-section">', unsafe_allow_html=True)

     # --- Detected tags line ---
    detected_classes = [
        label.replace('_', ' ').title()
        for label, score in predictions.items()
        if label != 'overall_toxic' and score > 0.5
    ]
    if detected_classes:
        st.markdown(
            " ".join([f"<span class='toxicity-badge1'>{cls}</span>" for cls in detected_classes]),
            unsafe_allow_html=True
        )
    st.markdown("### <span class='pastel-purple-highlight'>BERT Analysis</span>", unsafe_allow_html=True)
    
    # Overall toxicity
    overall_toxic = predictions['overall_toxic']
    is_toxic = overall_toxic > 0.5
    
    # Get detected classes above threshold
    detected_classes = []
    for label, score in predictions.items():
        if label != 'overall_toxic' and score > 0.5:
            formatted_label = label.replace('_', ' ').title()
            detected_classes.append(formatted_label)
    
    # Main results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Toxicity", 
            f"{overall_toxic:.1%}",
            delta="TOXIC" if is_toxic else "CLEAN",
            delta_color="inverse" if is_toxic else "normal"
        )
    
    with col2:
        if is_toxic:
            st.metric("Classification", "🚨 TOXIC")
            if detected_classes:
                st.markdown(f"""
               
                """, unsafe_allow_html=True)
        else:
            st.metric("Classification", "✅ CLEAN")
    
    with col3:
        toxic_categories = len(detected_classes)
        st.metric("Toxic Categories", f"{toxic_categories}/6")

    # Probability Distribution
    st.markdown("#### <span class='pastel-purple-highlight'>📈 Probability Distribution</span>", unsafe_allow_html=True)
    individual_scores = {k: v for k, v in predictions.items() if k != 'overall_toxic'}
    
    for tox_type, score in individual_scores.items():
        display_toxicity_bar_with_binary(tox_type, score, "🤖")

    # Radar Chart
    st.markdown("#### <span class='pastel-purple-highlight'>🎯 Toxicity Radar</span>", unsafe_allow_html=True)
    categories = [k.replace('_', ' ').title() for k in individual_scores.keys()]
    values = list(individual_scores.values())
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='BERT Scores',
        line=dict(color='#28a745', width=2),
        fillcolor='rgba(40, 167, 69, 0.3)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['0', '0.25', '0.5', '0.75', '1']
            )),
        showlegend=False,
        height=400,
        title="BERT Toxicity Radar Chart"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Toxicity Scores Bar Chart
    st.markdown("#### <span class='pastel-purple-highlight'>Toxicity Scores Bar Chart</span>", unsafe_allow_html=True)
    df_bert = pd.DataFrame({
        'Category': [k.replace('_', ' ').title() for k in individual_scores.keys()],
        'Score': list(individual_scores.values())
    })
    
    fig_bar = px.bar(
        df_bert, 
        y='Category', 
        x='Score',
        orientation='h',
        title="BERT Toxicity Scores",
        color='Score',
        color_continuous_scale=['#4CAF50', '#FFC107', '#ff4b4b'],
        range_color=[0, 1]
    )
    fig_bar.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)




def display_toxicity_bar_with_binary(tox_type, score, model_icon):
    """Display individual toxicity bar avec indication binaire"""
    is_toxic = score > 0.5
    color = "#ff4b4b" if is_toxic else "#4CAF50"
    icon = "🚨" if is_toxic else "✅"
    binary_label = "1" if is_toxic else "0"
    
    # Format label
    formatted_label = tox_type.replace('_', ' ').title()
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: bold;">{model_icon} {icon} {formatted_label}</span>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-weight: bold; color: {color}; font-size: 1.1rem;">{score:.1%}</span>
                <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 0.9rem;">
                    {binary_label}
                </span>
            </div>
        </div>
        <div class="toxicity-bar">
            <div class="toxicity-fill" style="width: {score*100}%; background: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_toxicity_bar(tox_type, score, model_icon):
    """Display individual toxicity bar"""
    is_toxic = score > 0.5
    color = "#ff4b4b" if is_toxic else "#4CAF50"
    icon = "🚨" if is_toxic else "✅"
    
    # Format label
    formatted_label = tox_type.replace('_', ' ').title()
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: bold;">{model_icon} {icon} {formatted_label}</span>
            <span style="font-weight: bold; color: {color};">{score:.1%}</span>
        </div>
        <div class="toxicity-bar">
            <div class="toxicity-fill" style="width: {score*100}%; background: {color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""

""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()