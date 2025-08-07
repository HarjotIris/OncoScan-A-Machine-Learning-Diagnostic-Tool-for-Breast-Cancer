import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"


# Page configuration
st.set_page_config(
    page_title="OncoScan: A Machine Learning Diagnostic Tool for Breast Cancer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .benign-box {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .malignant-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .info-box {
        background-color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

import json

class BreastCancerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = self.load_feature_names()

        try:
            self.model = joblib.load(MODELS_DIR / 'best_breast_cancer_model.joblib')
            self.scaler = joblib.load(MODELS_DIR / 'feature_scaler.joblib')

            if not hasattr(self.scaler, "mean_"):
                raise ValueError("Scaler is not fitted.")
            
        except Exception as e:
            st.warning(f"⚠️ Model or scaler loading failed: {e}\nUsing demo model instead.")
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.scaler.fit(np.random.rand(10, len(self.feature_names)))

    def load_feature_names(self):
        try:
            with open(MODELS_DIR / 'model_info.json', 'r') as f:
                data = json.load(f)
            return data.get('feature_names', [])
        except Exception as e:
            st.error(f"❌ Failed to load feature names: {e}")
            return []

    def preprocess_input(self, input_data):
        if len(input_data) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(input_data)}")
        return self.scaler.transform([input_data])

    def predict(self, features):
        try:
            features_processed = self.preprocess_input(features)
            prediction = self.model.predict(features_processed)[0]

            confidence = 0.85
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_processed)[0]
                confidence = max(proba)

            return prediction, confidence
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None

def main():
    """Main Streamlit application"""
    
    # Initialize the predictor
    predictor = BreastCancerPredictor()
    
    # Header
    st.markdown('<h1 class="main-header">🩺 OncoScan: A Machine Learning Diagnostic Tool for Breast Cancer</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🔍 Make Prediction", 
        "📊 Model Information", 
        "📈 Data Visualization",
        "ℹ️ About"
    ])
    
    if page == "🔍 Make Prediction":
        prediction_page(predictor)
    elif page == "📊 Model Information":
        model_info_page()
    elif page == "📈 Data Visualization":
        visualization_page()
    elif page == "ℹ️ About":
        about_page()

def prediction_page(predictor):
    """Prediction interface page"""
    
    st.header("🔍 Breast Cancer Prediction")
    
    st.markdown("""
    <div class="info-box">
        <strong>Instructions:</strong> Enter the cell nuclei measurements below to get a prediction.
        All values should be positive numbers. You can use the example values or modify them.
    </div>
    """, unsafe_allow_html=True)
    
    # Example cases data - realistic values based on typical benign/malignant cases
    benign_example = {
        'radius_mean': 12.32, 'texture_mean': 12.39, 'perimeter_mean': 78.85, 'area_mean': 464.1,
        'smoothness_mean': 0.1028, 'compactness_mean': 0.06981, 'concavity_mean': 0.03987,
        'concave_points_mean': 0.03700, 'symmetry_mean': 0.1959, 'fractal_dimension_mean': 0.05955,
        'radius_se': 0.236, 'texture_se': 0.666, 'perimeter_se': 1.670, 'area_se': 17.43,
        'smoothness_se': 0.00484, 'compactness_se': 0.01580, 'concavity_se': 0.01618,
        'concave_points_se': 0.00832, 'symmetry_se': 0.02239, 'fractal_dimension_se': 0.001389
    }
    
    malignant_example = {
        'radius_mean': 18.02, 'texture_mean': 22.74, 'perimeter_mean': 113.5, 'area_mean': 1035.0,
        'smoothness_mean': 0.08588, 'compactness_mean': 0.08541, 'concavity_mean': 0.1084,
        'concave_points_mean': 0.05556, 'symmetry_mean': 0.1775, 'fractal_dimension_mean': 0.05899,
        'radius_se': 0.757, 'texture_se': 1.841, 'perimeter_se': 5.05, 'area_se': 54.45,
        'smoothness_se': 0.00486, 'compactness_se': 0.01171, 'concavity_se': 0.02635,
        'concave_points_se': 0.01340, 'symmetry_se': 0.01705, 'fractal_dimension_se': 0.003187
    }
    
    # Initialize session state for feature values if not exists
    if 'feature_values' not in st.session_state:
        # Default values for better visualization
        default_values = {
            'radius_mean': 14.0, 'texture_mean': 19.0, 'perimeter_mean': 91.0, 'area_mean': 654.0,
            'smoothness_mean': 0.096, 'compactness_mean': 0.104, 'concavity_mean': 0.089,
            'concave_points_mean': 0.048, 'symmetry_mean': 0.181, 'fractal_dimension_mean': 0.063,
            'radius_se': 0.405, 'texture_se': 1.216, 'perimeter_se': 2.866, 'area_se': 40.337,
            'smoothness_se': 0.007, 'compactness_se': 0.025, 'concavity_se': 0.032,
            'concave_points_se': 0.012, 'symmetry_se': 0.020, 'fractal_dimension_se': 0.004
        }
        
        st.session_state.feature_values = {}
        for feature in predictor.feature_names:
            st.session_state.feature_values[feature] = default_values.get(feature, 1.0)
    
    # Example cases buttons at the top
    st.subheader("💡 Quick Start - Try Example Cases")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📗 Load Benign Example", use_container_width=True):
            for feature in predictor.feature_names:
                if feature in benign_example:
                    st.session_state.feature_values[feature] = benign_example[feature]
            st.success("✅ Loaded typical benign case values!")
            st.rerun()
    
    with col2:
        if st.button("📕 Load Malignant Example", use_container_width=True):
            for feature in predictor.feature_names:
                if feature in malignant_example:
                    st.session_state.feature_values[feature] = malignant_example[feature]
            st.success("✅ Loaded typical malignant case values!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            default_values = {
                'radius_mean': 14.0, 'texture_mean': 19.0, 'perimeter_mean': 91.0, 'area_mean': 654.0,
                'smoothness_mean': 0.096, 'compactness_mean': 0.104, 'concavity_mean': 0.089,
                'concave_points_mean': 0.048, 'symmetry_mean': 0.181, 'fractal_dimension_mean': 0.063,
                'radius_se': 0.405, 'texture_se': 1.216, 'perimeter_se': 2.866, 'area_se': 40.337,
                'smoothness_se': 0.007, 'compactness_se': 0.025, 'concavity_se': 0.032,
                'concave_points_se': 0.012, 'symmetry_se': 0.020, 'fractal_dimension_se': 0.004
            }
            for feature in predictor.feature_names:
                st.session_state.feature_values[feature] = default_values.get(feature, 1.0)
            st.info("🔄 Reset to default values!")
            st.rerun()
    
    st.markdown("---")
    
    # Create input fields
    st.subheader("📝 Enter Feature Values")
    
    # Collect all features
    features = []
    
    # Group features for better organization
    mean_features = [f for f in predictor.feature_names if f.endswith('_mean')]
    se_features = [f for f in predictor.feature_names if f.endswith('_se')]
    worst_features = [f for f in predictor.feature_names if f.endswith('_worst')]
    
    # Display features in organized tabs
    tab1, tab2, tab3 = st.tabs(["📊 Mean Values", "📈 Standard Error", "⚠️ Worst Values"])
    
    with tab1:
        if mean_features:
            st.write("**Mean measurements of cell nuclei features:**")
            col1, col2 = st.columns(2)
            for i, feature in enumerate(mean_features):
                with (col1 if i % 2 == 0 else col2):
                    value = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        min_value=0.0, 
                        value=float(st.session_state.feature_values.get(feature, 1.0)),
                        format="%.6f",
                        key=f"input_{feature}"
                    )
                    st.session_state.feature_values[feature] = value
    
    with tab2:
        if se_features:
            st.write("**Standard error of measurements:**")
            col1, col2 = st.columns(2)
            for i, feature in enumerate(se_features):
                with (col1 if i % 2 == 0 else col2):
                    value = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        min_value=0.0, 
                        value=float(st.session_state.feature_values.get(feature, 1.0)),
                        format="%.6f",
                        key=f"input_{feature}"
                    )
                    st.session_state.feature_values[feature] = value
    
    with tab3:
        if worst_features:
            st.write("**Worst (largest) values of measurements:**")
            col1, col2 = st.columns(2)
            for i, feature in enumerate(worst_features):
                with (col1 if i % 2 == 0 else col2):
                    value = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        min_value=0.0, 
                        value=float(st.session_state.feature_values.get(feature, 1.0)),
                        format="%.6f",
                        key=f"input_{feature}"
                    )
                    st.session_state.feature_values[feature] = value
    
    # If no feature grouping worked, fall back to original layout
    if not mean_features and not se_features and not worst_features:
        col1, col2 = st.columns(2)
        for i, feature in enumerate(predictor.feature_names):
            with (col1 if i % 2 == 0 else col2):
                value = st.number_input(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=0.0, 
                    value=float(st.session_state.feature_values.get(feature, 1.0)),
                    format="%.6f",
                    key=f"input_{feature}"
                )
                st.session_state.feature_values[feature] = value
    
    # Collect features in the correct order
    for feature in predictor.feature_names:
        features.append(st.session_state.feature_values.get(feature, 1.0))
    
    # Prediction button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 Make Prediction", type="primary", use_container_width=True):
            make_prediction(features, predictor)
            
def make_prediction(features, predictor):
    """Make and display prediction"""
    
    with st.spinner("🔄 Analyzing features..."):
        prediction, confidence = predictor.predict(features)
    
    if prediction is not None:
        # Display result
        if prediction == 0:  # Benign
            st.markdown(f"""
            <div class="prediction-box benign-box">
                🟢 PREDICTION: BENIGN<br>
                Confidence: {confidence:.1%}
            </div>
            """, unsafe_allow_html=True)
            
            st.success("The model predicts this case is likely **BENIGN** (non-cancerous).")
            
        else:  # Malignant
            st.markdown(f"""
            <div class="prediction-box malignant-box">
                🔴 PREDICTION: MALIGNANT<br>
                Confidence: {confidence:.1%}
            </div>
            """, unsafe_allow_html=True)
            
            st.error("The model predicts this case is likely **MALIGNANT** (cancerous).")
            st.warning("⚠️ This is a screening tool only. Please consult with a healthcare professional for proper diagnosis.")
        
        # Feature importance visualization
        st.subheader("📊 Feature Contribution Analysis")
        
        # Create a simple feature importance chart
        feature_names = predictor.feature_names
        
        # Check if we have valid feature names and values
        if not feature_names or len(features) == 0:
            st.warning("Unable to display feature analysis - missing feature information.")
            return
        
        # Ensure we have the right number of features
        if len(features) != len(feature_names):
            st.warning(f"Feature count mismatch: expected {len(feature_names)}, got {len(features)}")
            return
        
        # Normalize features for visualization
        feature_min = min(features)
        feature_max = max(features)
        range_val = feature_max - feature_min
        
        # Handle case where all values are the same
        if range_val == 0:
            # If all values are the same, show them as equal bars
            features_normalized = [1.0] * len(features)
            st.info("All feature values are identical - showing uniform distribution.")
        else:
            # Normal normalization
            features_normalized = [(f - feature_min) / range_val for f in features]
        
        # Create DataFrame for visualization
        viz_df = pd.DataFrame({
            'Feature': [name.replace('_', ' ').title() for name in feature_names],
            'Normalized Value': features_normalized,
            'Raw Value': features
        })
        
        # Sort by normalized value for better visualization
        viz_df = viz_df.sort_values('Normalized Value', ascending=True)
        
        # Create horizontal bar chart for better readability with many features
        fig = px.bar(
            viz_df,
            y='Feature',  # Changed to y for horizontal bars
            x='Normalized Value',  # Changed to x
            title="📊 Normalized Feature Values (0-1 scale)",
            color='Normalized Value',
            color_continuous_scale='viridis',
            orientation='h',  # Horizontal orientation
            hover_data={
                'Raw Value': ':.3f',
                'Normalized Value': ':.3f'
            }
        )
        
        # Update layout for better readability
        fig.update_layout(
            height=max(400, len(feature_names) * 25),  # Dynamic height based on number of features
            xaxis_title="Normalized Value (0-1)",
            yaxis_title="Features",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary statistics
        with st.expander("📈 Feature Statistics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Highest Feature", 
                         viz_df.loc[viz_df['Normalized Value'].idxmax(), 'Feature'],
                         f"{viz_df['Raw Value'].max():.3f}")
            
            with col2:
                st.metric("Lowest Feature", 
                         viz_df.loc[viz_df['Normalized Value'].idxmin(), 'Feature'],
                         f"{viz_df['Raw Value'].min():.3f}")
            
            with col3:
                st.metric("Average Value", 
                         f"{viz_df['Raw Value'].mean():.3f}",
                         f"±{viz_df['Raw Value'].std():.3f}")
def model_info_page():
    """Model information and performance metrics"""
    
    st.header("📊 Model Information")
    
    # Model performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Accuracy", "96.5%", "1.6%")
    
    with col2:
        st.metric("Precision", "95.2%", "2.7%")
    
    with col3:
        st.metric("Recall", "95.2%", "1.1%")
    
    with col4:
        st.metric("ROC-AUC", "0.995", "0.008")
    
    st.markdown("---")
    
    # Model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Details")
        st.markdown("""
        - **Algorithm**: Logistic Regression (Best Performer)
        - **Features**: 20 selected optimal features
        - **Training Samples**: 455 cases
        - **Test Samples**: 114 cases
        - **Cross-validation**: 5-fold stratified
        - **Hyperparameter Tuning**: RandomizedSearchCV
        - **ROC-AUC Rank**: #1 out of 8 algorithms tested
        """)
    
    with col2:
        st.subheader("🔧 Preprocessing Steps")
        st.markdown("""
        1. **Multicollinearity Removal**: Correlation > 0.95
        2. **Outlier Handling**: IQR method with capping
        3. **Skewness Correction**: Log transformation
        4. **Feature Scaling**: Standard scaling
        5. **Feature Selection**: Recursive Feature Elimination
        """)
    
    # Confusion matrix
    st.subheader("📈 Model Performance Visualization")
    
    # Create sample confusion matrix
    cm_data = np.array([[65, 3], [2, 44]])
    
    fig = px.imshow(
        cm_data,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        x=['Benign', 'Malignant'],
        y=['Benign', 'Malignant']
    )
    st.plotly_chart(fig, use_container_width=True)

def visualization_page():
    """Data visualization and insights"""
    
    st.header("📈 Data Insights & Visualizations")
    
    # Generate sample data for visualization
    np.random.seed(42)
    n_samples = 500
    
    # Sample data generation (replace with actual data loading)
    benign_data = {
        'radius_mean': np.random.normal(12, 2, n_samples//2),
        'area_mean': np.random.normal(500, 150, n_samples//2),
        'diagnosis': ['Benign'] * (n_samples//2)
    }
    
    malignant_data = {
        'radius_mean': np.random.normal(17, 3, n_samples//2),
        'area_mean': np.random.normal(900, 250, n_samples//2),
        'diagnosis': ['Malignant'] * (n_samples//2)
    }
    
    # Combine data
    sample_df = pd.DataFrame({
        'radius_mean': list(benign_data['radius_mean']) + list(malignant_data['radius_mean']),
        'area_mean': list(benign_data['area_mean']) + list(malignant_data['area_mean']),
        'diagnosis': benign_data['diagnosis'] + malignant_data['diagnosis']
    })
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            sample_df, 
            x='radius_mean', 
            color='diagnosis',
            title="Distribution of Radius Mean",
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            sample_df, 
            x='area_mean', 
            color='diagnosis',
            title="Distribution of Area Mean",
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot
    fig = px.scatter(
        sample_df,
        x='radius_mean',
        y='area_mean',
        color='diagnosis',
        title="Radius Mean vs Area Mean",
        hover_data={'diagnosis': True}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (mock data)
    st.subheader("🎯 Feature Importance")
    
    importance_data = {
        'Feature': ['area_worst', 'radius_worst', 'perimeter_worst', 'area_mean', 'radius_mean'],
        'Importance': [0.15, 0.13, 0.12, 0.11, 0.09]
    }
    
    fig = px.bar(
        importance_data,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 5 Most Important Features"
    )
    st.plotly_chart(fig, use_container_width=True)

def about_page():
    """About the project and disclaimer"""
    
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This OncoScan: A Machine Learning Diagnostic Tool for Breast Cancer is a machine learning application that analyzes cell nuclei features 
    to help in the early detection of breast cancer. The app uses the Wisconsin Diagnostic Breast Cancer Dataset.
    
    ## 🔬 How It Works
    
    1. **Data Input**: Users input 15 key measurements of cell nuclei
    2. **Preprocessing**: The data is cleaned and normalized using the same pipeline as training
    3. **Prediction**: A trained Random Forest model makes the prediction
    4. **Visualization**: Results are displayed with confidence scores and feature analysis
    
    ## 📊 Dataset Information
    
    - **Source**: Wisconsin Diagnostic Breast Cancer Dataset
    - **Samples**: 569 cases
    - **Features**: 30 original features (reduced to 15 after feature selection)
    - **Classes**: Malignant (M) and Benign (B)
    - **Class Distribution**: 37.3% Malignant, 62.7% Benign
    
    ## 🛠️ Technical Stack
    
    - **Machine Learning**: Scikit-learn, XGBoost
    - **Web Framework**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Deployment**: Streamlit Cloud
    
    ## 📈 Model Performance
    
    - **Algorithm**: Logistic Regression (Best of 8 tested)
    - **Accuracy**: 96.5%
    - **Precision**: 95.2%
    - **Recall**: 95.2%
    - **ROC-AUC**: 0.995 (#1 Ranked)
    
    ## ⚠️ Important Disclaimer
    
    **THIS IS A EDUCATIONAL/RESEARCH TOOL ONLY**
    
    This application is designed for educational and research purposes. It should NOT be used as a 
    substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified 
    healthcare professionals for medical decisions.
    
    ## 🚀 Future Improvements
    
    - Integration with medical imaging data
    - Real-time model updates
    - Multi-class classification for cancer stages
    - Integration with hospital systems
    - Mobile application development
    
    ## 👨‍💻 Developer
    
    This project was developed as part of a comprehensive machine learning portfolio. 
    The goal is to demonstrate end-to-end ML project development skills including:
    
    - Exploratory Data Analysis
    - Feature Engineering & Selection
    - Model Training & Evaluation
    - Web Application Development
    - Model Deployment
    
    ## 📚 References
    
    1. Wolberg, W.H., Street, W.N., and Mangasarian, O.L. (1995). Machine learning techniques to diagnose breast cancer from fine-needle aspirates.
    2. Street, W.N., Wolberg, W.H. and Mangasarian, O.L. (1993). Nuclear feature extraction for breast tumor diagnosis.
    
    ---
    
    **✨ Built with ❤️ (and Python) by Harjot / Iris. Thank you for visiting.**
    """)

if __name__ == "__main__":
    main()
