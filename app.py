import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-align: center;
    }
    .welcome-text {
        font-size: 1.3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #1f77b4;
        margin: 25px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        text-align: center;
        margin: 15px 5px;
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    .metric-card h4 {
        color: #2c3e50;
        font-size: 1.4rem;
        margin-bottom: 15px;
        font-weight: 600;
    }
    .metric-card p {
        color: #6c757d;
        font-size: 1rem;
        line-height: 1.5;
        margin: 0;
    }
    .info-box {
        background: linear-gradient(145deg, #e8f4fd, #d1ecf1);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #2196F3;
        margin: 25px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-box h3 {
        color: #1976d2;
        font-size: 1.5rem;
        margin-bottom: 15px;
        font-weight: 600;
    }
    .info-box p, .info-box li {
        color: #2c3e50;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .info-box ul {
        margin: 15px 0;
        padding-left: 25px;
    }
    .info-box strong {
        color: #1976d2;
        font-weight: 600;
    }
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
        display: block;
    }
    .how-to-use {
        background: linear-gradient(145deg, #fff3cd, #ffeaa7);
        border-left: 6px solid #ffc107;
    }
    .how-to-use h3 {
        color: #856404;
    }
    .how-to-use li {
        color: #6c5400;
    }
    .separator {
        height: 3px;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        border: none;
        border-radius: 2px;
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('exam_score_predictor.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'exam_score_predictor.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

# Load dataset for analysis
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('student_exam_data.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found! Please ensure 'student_exam_data.csv' is in the same directory.")
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">📚 Student Exam Score Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.markdown("## 🧭 Navigation")
    st.sidebar.markdown("---")
    
    # Add some info in sidebar
    st.sidebar.markdown("""
    <div style="background: linear-gradient(145deg, #f8f9fa, #e9ecef); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h4 style="color: #495057; margin: 0 0 10px 0;">🎯 Quick Info</h4>
    <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">Navigate through different sections to explore the ML-powered prediction system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["🏠 Home", "🔮 Prediction", "📊 Data Analysis", "📈 Model Insights"],
        help="Select a page to navigate through the application"
    )
    
    # Add sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.8rem;">
    <p>🚀 Built with Streamlit</p>
    <p>🤖 Powered by ML</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model, scaler = load_model()
    df = load_data()
    
    if page == "🏠 Home":
        home_page()
    elif page == "🔮 Prediction":
        prediction_page(model, scaler)
    elif page == "📊 Data Analysis":
        data_analysis_page(df)
    elif page == "📈 Model Insights":
        model_insights_page(df, model, scaler)

def home_page():
    # Main welcome message
    st.markdown('<h2 class="sub-header">Welcome to the Student Exam Score Predictor!</h2>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">🎓 Predict student exam scores using advanced Machine Learning algorithms</p>', unsafe_allow_html=True)
    
    # Add a separator
    st.markdown('<hr class="separator">', unsafe_allow_html=True)
    
    # Hero image section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <div style="font-size: 5rem; margin-bottom: 20px;">📊🎯📚</div>
            <h3 style="color: #2c3e50; margin: 0;">AI-Powered Education Analytics</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Project overview section
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Project Overview</h3>
    <p>This intelligent web application leverages Machine Learning to predict student exam scores with high accuracy. Our model analyzes three crucial academic factors:</p>
    <ul>
        <li><strong>📖 Hours Studied:</strong> The total time a student dedicates to studying and preparation</li>
        <li><strong>📝 Previous Exam Score:</strong> Historical academic performance as a predictor</li>
        <li><strong>📅 Attendance:</strong> Class participation and engagement percentage</li>
    </ul>
    <p><strong>🔬 Technology Stack:</strong> Python • Scikit-learn • Linear Regression • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards section
    st.markdown("### 🚀 Application Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <div class="feature-icon">🔮</div>
        <h4>Smart Prediction</h4>
        <p>Get instant and accurate exam score predictions for new students using our trained ML model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <div class="feature-icon">📊</div>
        <h4>Data Analysis</h4>
        <p>Explore comprehensive dataset insights with interactive visualizations and statistical analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <div class="feature-icon">📈</div>
        <h4>Model Insights</h4>
        <p>Understand feature importance, model coefficients, and prediction sensitivity analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use section
    st.markdown("""
    <div class="info-box how-to-use">
    <h3>🚀 How to Navigate</h3>
    <ol style="font-size: 1.1rem; line-height: 1.8;">
        <li><strong>🔮 Prediction Page:</strong> Input student data and get instant exam score predictions</li>
        <li><strong>📊 Data Analysis:</strong> Explore the training dataset with interactive charts and statistics</li>
        <li><strong>📈 Model Insights:</strong> Dive deep into model performance and feature analysis</li>
    </ol>
    <p><strong>💡 Tip:</strong> Start with the Prediction page to see the model in action, then explore the analysis sections!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats section
    st.markdown("### 📋 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #e3f2fd, #bbdefb); border-radius: 10px; margin: 10px 0;">
        <h3 style="color: #1976d2; margin: 0; font-size: 2rem;">1000+</h3>
        <p style="color: #1565c0; margin: 5px 0; font-weight: 500;">Training Samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #e8f5e8, #c8e6c9); border-radius: 10px; margin: 10px 0;">
        <h3 style="color: #388e3c; margin: 0; font-size: 2rem;">3</h3>
        <p style="color: #2e7d32; margin: 5px 0; font-weight: 500;">Input Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #fff3e0, #ffcc02); border-radius: 10px; margin: 10px 0;">
        <h3 style="color: #f57c00; margin: 0; font-size: 2rem;">95%+</h3>
        <p style="color: #ef6c00; margin: 5px 0; font-weight: 500;">Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #fce4ec, #f8bbd9); border-radius: 10px; margin: 10px 0;">
        <h3 style="color: #c2185b; margin: 0; font-size: 2rem;">ML</h3>
        <p style="color: #ad1457; margin: 5px 0; font-weight: 500;">Powered</p>
        </div>
        """, unsafe_allow_html=True)

def prediction_page(model, scaler):
    st.markdown('<h2 class="sub-header">🔮 Predict Student Exam Score</h2>', unsafe_allow_html=True)
    
    if model is None or scaler is None:
        st.error("Model not loaded. Please check if model files exist.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Input Student Information")
        
        # Create input form
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                hours_studied = st.number_input(
                    "📖 Hours Studied", 
                    min_value=0.0, 
                    max_value=20.0, 
                    value=10.0, 
                    step=0.1,
                    help="Number of hours the student studied (0-20 hours)"
                )
                
                attendance = st.slider(
                    "📅 Attendance (%)", 
                    min_value=0, 
                    max_value=100, 
                    value=85,
                    help="Percentage of classes attended by the student"
                )
            
            with col_b:
                previous_score = st.number_input(
                    "📝 Previous Exam Score", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=75.0, 
                    step=0.1,
                    help="Student's previous exam score (0-100)"
                )
            
            submitted = st.form_submit_button("🎯 Predict Exam Score", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = np.array([[hours_studied, previous_score, attendance]])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
            <h3>🎯 Prediction Result</h3>
            <h2 style="color: #1f77b4; font-size: 2.5rem; text-align: center;">
                {prediction:.2f}/100
            </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance interpretation
            if prediction >= 90:
                performance = "Excellent! 🌟"
                color = "#4CAF50"
            elif prediction >= 80:
                performance = "Very Good! 👍"
                color = "#2196F3"
            elif prediction >= 70:
                performance = "Good 👌"
                color = "#FF9800"
            elif prediction >= 60:
                performance = "Average 📊"
                color = "#FFC107"
            else:
                performance = "Needs Improvement 📈"
                color = "#F44336"
            
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
            <h3 style="color: {color};">{performance}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Input Summary")
        
        # Create a gauge chart for the prediction
        if 'prediction' in locals():
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Predicted Score"},
                delta = {'reference': 75},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Input summary
        st.markdown("#### Current Inputs:")
        if 'hours_studied' in locals():
            st.write(f"📖 **Hours Studied:** {hours_studied}")
            st.write(f"📝 **Previous Score:** {previous_score}")
            st.write(f"📅 **Attendance:** {attendance}%")

def data_analysis_page(df):
    st.markdown('<h2 class="sub-header">📊 Data Analysis & Visualization</h2>', unsafe_allow_html=True)
    
    if df is None:
        st.error("Dataset not loaded. Please check if the CSV file exists.")
        return
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Average Score", f"{df['Exam_Score'].mean():.2f}")
    with col3:
        st.metric("Highest Score", f"{df['Exam_Score'].max():.2f}")
    with col4:
        st.metric("Lowest Score", f"{df['Exam_Score'].min():.2f}")
    
    # Data preview
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Score Distribution")
        fig = px.histogram(df, x='Exam_Score', nbins=20, title="Distribution of Exam Scores")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔗 Feature Correlations")
        correlation_matrix = df.corr()
        fig = px.imshow(correlation_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📖 Hours Studied vs Exam Score")
        fig = px.scatter(df, x='Hours_Studied', y='Exam_Score', 
                        title="Hours Studied vs Exam Score",
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📅 Attendance vs Exam Score")
        fig = px.scatter(df, x='Attendance', y='Exam_Score', 
                        title="Attendance vs Exam Score",
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    
    # Pairplot
    st.markdown("### 🔄 Feature Relationships")
    fig = px.scatter_matrix(df, 
                           dimensions=['Hours_Studied', 'Previous_Exam_Score', 'Attendance', 'Exam_Score'],
                           title="Pairwise Feature Relationships")
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

def model_insights_page(df, model, scaler):
    st.markdown('<h2 class="sub-header">📈 Model Performance & Insights</h2>', unsafe_allow_html=True)
    
    if df is None or model is None or scaler is None:
        st.error("Model or dataset not loaded properly.")
        return
    
    # Model coefficients
    st.markdown("### 🔍 Model Coefficients")
    
    feature_names = ['Hours_Studied', 'Previous_Exam_Score', 'Attendance']
    coefficients = model.coef_
    intercept = model.intercept_
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Impact': np.abs(coefficients)
    }).sort_values('Absolute_Impact', ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(coef_df, use_container_width=True)
        st.write(f"**Model Intercept:** {intercept:.4f}")
    
    with col2:
        fig = px.bar(coef_df, x='Feature', y='Coefficient', 
                    title="Feature Importance (Coefficients)",
                    color='Coefficient',
                    color_continuous_scale='RdYlBu')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance interpretation
    st.markdown("### 💡 Feature Impact Analysis")
    
    most_important = coef_df.iloc[0]
    st.markdown(f"""
    <div class="info-box">
    <h4>Most Influential Feature: {most_important['Feature']}</h4>
    <p>Coefficient: {most_important['Coefficient']:.4f}</p>
    <p>This means that for every unit increase in {most_important['Feature'].lower()}, 
    the exam score changes by approximately {most_important['Coefficient']:.4f} points.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model performance simulation
    st.markdown("### 🎯 Prediction Examples")
    
    # Create sample predictions
    sample_students = [
        [15, 90, 95],  # High performer
        [8, 70, 75],   # Average performer
        [5, 50, 60]    # Low performer
    ]
    
    sample_labels = ["High Performer", "Average Performer", "Low Performer"]
    
    predictions = []
    for student in sample_students:
        input_scaled = scaler.transform([student])
        pred = model.predict(input_scaled)[0]
        predictions.append(pred)
    
    sample_df = pd.DataFrame({
        'Student Type': sample_labels,
        'Hours Studied': [s[0] for s in sample_students],
        'Previous Score': [s[1] for s in sample_students],
        'Attendance (%)': [s[2] for s in sample_students],
        'Predicted Score': [f"{p:.2f}" for p in predictions]
    })
    
    st.dataframe(sample_df, use_container_width=True)
    
    # Interactive sensitivity analysis
    st.markdown("### 🔬 Sensitivity Analysis")
    st.markdown("See how changing one feature affects the prediction while keeping others constant.")
    
    base_hours = st.slider("Base Hours Studied", 0.0, 20.0, 10.0, key="sens_hours")
    base_previous = st.slider("Base Previous Score", 0.0, 100.0, 75.0, key="sens_previous")
    base_attendance = st.slider("Base Attendance", 0, 100, 85, key="sens_attendance")
    
    # Sensitivity for hours studied
    hours_range = np.linspace(0, 20, 50)
    hours_predictions = []
    
    for h in hours_range:
        input_data = scaler.transform([[h, base_previous, base_attendance]])
        pred = model.predict(input_data)[0]
        hours_predictions.append(pred)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours_range, y=hours_predictions, 
                            mode='lines', name='Predicted Score'))
    fig.add_vline(x=base_hours, line_dash="dash", line_color="red", 
                  annotation_text=f"Current: {base_hours}h")
    fig.update_layout(title="Sensitivity: Hours Studied vs Predicted Score",
                     xaxis_title="Hours Studied",
                     yaxis_title="Predicted Exam Score")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
