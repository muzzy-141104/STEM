import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import random
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Soil Erosion Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {border-radius: 4px; padding: 8px 16px;}
    .stSelectbox, .stNumberInput, .stSlider {margin-bottom: 15px;}
    .prediction-card {border-radius: 8px; padding: 20px; margin-bottom: 20px; 
                      box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .best-model {border-left: 5px solid #4CAF50; background-color: #f8fff8;}
    .feature-importance-plot {margin-top: 20px;}
    .header-text {color: #2c3e50; font-weight: 700;}
    .subheader-text {color: #3498db; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# Load models and scaler
@st.cache_resource
def load_models():
    models = {
        "Random Forest": joblib.load("soil_erosion_model_random_forest.joblib"),
        "XGBoost": joblib.load("soil_erosion_model_xgboost.joblib"),
        "Gradient Boosting": joblib.load("soil_erosion_model_gradient_boosting.joblib"),
        "Neural Network": joblib.load("soil_erosion_model_neural_network.joblib")
    }
    scaler = joblib.load("soil_erosion_scaler.joblib")
    return models, scaler

models, scaler = load_models()

# Model performance data
model_performance = {
    "Random Forest": {"R2": 0.54, "RMSE": 0.035, "color": "#3498db"},
    "XGBoost": {"R2": 0.52, "RMSE": 0.036, "color": "#e74c3c"},
    "Gradient Boosting": {"R2": 0.53, "RMSE": 0.035, "color": "#9b59b6"},
    "Neural Network": {"R2": 0.41, "RMSE": 0.039, "color": "#f39c12"}
}

# Visualization data
visualizations = {
    "Basic Analysis": [
        ('erosion_distribution.png', 'Erosion Distribution'),
        ('correlation_heatmap.png', 'Feature Correlation Heatmap'),
        ('erosion_by_landuse.png', 'Erosion by Land Use Type')
    ],
    "Enhanced Visualizations": [
        ('top_features_scatter.png', 'Top Features vs Erosion'),
        ('feature_distributions.png', 'Feature Distributions'),
        ('enhanced_correlation.png', 'Enhanced Correlation Matrix'),
        ('land_use_boxplot.png', 'Land Use Impact Boxplot'),
        ('geographic_distribution.png', 'Geographic Distribution'),
        ('enhanced_feature_importance.png', 'Feature Importance'),
        ('enhanced_residual_plot.png', 'Model Residuals')
    ],
    "Model Performance": [
        ('model_performance.png', 'Model Comparison'),
        ('model_errors.png', 'Model Error Analysis')
    ],
    "Feature Importance": [
        ('feature_importance_random_forest.png', 'Random Forest'),
        ('feature_importance_xgboost.png', 'XGBoost'),
        ('feature_importance_gradient_boosting.png', 'Gradient Boosting'),
        ('feature_importance_neural_network.png', 'Neural Network')
    ],
    "Prediction Accuracy": [
        ('prediction_accuracy_random_forest.png', 'Random Forest'),
        ('prediction_accuracy_xgboost.png', 'XGBoost'),
        ('prediction_accuracy_gradient_boosting.png', 'Gradient Boosting'),
        ('prediction_accuracy_neural_network.png', 'Neural Network')
    ]
}

# Feature information
feature_info = {
    'slope_angle_deg': {'name': 'Slope Angle', 'unit': 'degrees', 'min': 0, 'max': 60, 'step': 1, 'default': 15},
    'NDVI': {'name': 'NDVI', 'unit': 'index', 'min': -0.1, 'max': 0.9, 'step': 0.01, 'default': 0.5},
    'rainfall_erosivity': {'name': 'Rainfall Erosivity', 'unit': 'index', 'min': 100, 'max': 10000, 'step': 100, 'default': 3000},
    'soil_moisture_percent': {'name': 'Soil Moisture', 'unit': '%', 'min': 5, 'max': 95, 'step': 1, 'default': 30},
    'elevation_m': {'name': 'Elevation', 'unit': 'meters', 'min': 0, 'max': 4000, 'step': 10, 'default': 500},
    'slope_length': {'name': 'Slope Length', 'unit': 'meters', 'min': 10, 'max': 1000, 'step': 10, 'default': 100},
    'aspect_deg': {'name': 'Aspect', 'unit': 'degrees', 'min': 0, 'max': 360, 'step': 5, 'default': 180},
    'plan_curvature': {'name': 'Plan Curvature', 'unit': 'index', 'min': -0.3, 'max': 0.3, 'step': 0.01, 'default': 0},
    'profile_curvature': {'name': 'Profile Curvature', 'unit': 'index', 'min': -0.5, 'max': 0.5, 'step': 0.01, 'default': 0},
    'soil_organic_matter': {'name': 'Soil Organic Matter', 'unit': '%', 'min': 0.5, 'max': 5.0, 'step': 0.1, 'default': 2.0},
    'soil_depth': {'name': 'Soil Depth', 'unit': 'meters', 'min': 0.5, 'max': 2.0, 'step': 0.1, 'default': 1.0}
}

# Model explanations
model_explanations = {
    "Random Forest": """
    Random Forest is recommended because it consistently achieves the highest accuracy (R²) and lowest error (RMSE) on soil erosion data. 
    It handles complex, nonlinear relationships between features, is robust to noise and outliers, and provides clear feature importance. 
    Random Forest is especially effective for tabular environmental data with many interacting variables.
    """,
    "XGBoost": """
    XGBoost is a powerful boosting algorithm that can sometimes outperform Random Forest with careful tuning. 
    It is robust and handles missing data well, but in most cases, its accuracy is slightly lower than Random Forest. 
    It is still a strong alternative, especially for large datasets.
    """,
    "Gradient Boosting": """
    Gradient Boosting is similar to XGBoost and can model complex relationships. 
    It performs nearly as well as Random Forest in most results, but is a bit more sensitive to overfitting and requires more tuning.
    """,
    "Neural Network": """
    Neural Networks are powerful for very large and complex datasets, but for soil erosion prediction, they often underperform compared to tree-based models. 
    They require more data and tuning, and are less interpretable for tabular environmental data.
    """
}

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a section", 
                          ["📊 Visualizations", "🔮 Prediction", "📚 About"])

if app_mode == "📊 Visualizations":
    # Visualization explorer
    st.title("🌱 Soil Erosion Visualization Explorer")
    st.markdown("Explore various visualizations of soil erosion data and model performance.")
    
    # Category selection
    category = st.selectbox("Select a visualization category", list(visualizations.keys()))
    
    # Visualization selection
    viz_options = [viz[1] for viz in visualizations[category]]
    selected_viz = st.selectbox("Select visualization", viz_options)
    
    # Find the selected visualization file
    viz_file = None
    for viz in visualizations[category]:
        if viz[1] == selected_viz:
            viz_file = viz[0]
            break
    
    # Display visualization
    if viz_file and os.path.exists(viz_file):
        st.image(viz_file, use_column_width=True)
        st.caption(f"Figure: {selected_viz}")
    else:
        st.warning(f"Visualization file not found: {viz_file}")

elif app_mode == "🔮 Prediction":
    # Prediction interface
    st.title("🔮 Soil Erosion Prediction")
    st.markdown("Predict soil erosion rates based on environmental factors.")
    
    with st.expander("⚙️ Input Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        # Feature inputs
        feature_values = {}
        for i, (feature, info) in enumerate(feature_info.items()):
            col = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3
            with col:
                # Determine if this should be float or int
                is_float = any(isinstance(info[k], float) or (isinstance(info[k], int) and '.' in str(info[k])) for k in ['min', 'max', 'step', 'default'])
                if is_float or info['step'] < 1:
                    feature_values[feature] = st.slider(
                        f"{info['name']} ({info['unit']})",
                        min_value=float(info['min']),
                        max_value=float(info['max']),
                        value=float(info['default']),
                        step=float(info['step']),
                        help=f"Range: {info['min']} to {info['max']} {info['unit']}"
                    )
                else:
                    feature_values[feature] = st.slider(
                        f"{info['name']} ({info['unit']})",
                        min_value=int(info['min']),
                        max_value=int(info['max']),
                        value=int(info['default']),
                        step=int(info['step']),
                        help=f"Range: {info['min']} to {info['max']} {info['unit']}"
                    )
    
    # Additional parameters
    with st.expander("📏 Area and Time Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.number_input("Area (hectares)", min_value=0.1, max_value=1000.0, value=1.0, step=0.1)
        with col2:
            months = st.number_input("Time Period (months)", min_value=1, max_value=120, value=12, step=1)
        with col3:
            model_choice = st.selectbox("Model", ["All Models"] + list(models.keys()))
    
    # Action buttons
    col1, col2, col3 = st.columns([1,1,3])
    with col1:
        predict_btn = st.button("🚀 Predict Erosion")
    with col2:
        if st.button("🎲 Randomize Inputs"):
            for feature, info in feature_info.items():
                if 'curvature' in feature:
                    # Use normal distribution centered at 0 for curvature features
                    val = np.random.normal(0, (info['max']-info['min'])/6)
                    val = max(min(val, info['max']), info['min'])
                else:
                    val = random.uniform(info['min'], info['max'])
                
                if feature in ['NDVI', 'plan_curvature', 'profile_curvature']:
                    val = round(val, 2)
                elif feature in ['soil_organic_matter', 'soil_depth']:
                    val = round(val, 1)
                else:
                    val = int(val) if feature not in ['NDVI'] else round(val, 2)
                
                feature_values[feature] = val
    
    # Prediction results
    if predict_btn:
        st.subheader("📊 Prediction Results")
        
        # Prepare input data
        features = [feature_values[f] for f in feature_info.keys()]
        features_df = pd.DataFrame([features], columns=list(feature_info.keys()))
        X_scaled = scaler.transform(features_df)
        
        # Make predictions
        results = {}
        for model_name, model in models.items():
            if model_choice != "All Models" and model_choice != model_name:
                continue
            
            pred_annual = model.predict(X_scaled)[0]
            pred_period = pred_annual * (months / 12)
            total_loss = pred_period * area
            
            results[model_name] = {
                "annual": pred_annual,
                "period": pred_period,
                "total": total_loss,
                "R2": model_performance[model_name]["R2"],
                "RMSE": model_performance[model_name]["RMSE"],
                "color": model_performance[model_name]["color"]
            }
        
        # Find best model
        if results:
            best_model = max(results, key=lambda k: results[k]["R2"])
            
            # Display results in cards
            cols = st.columns(len(results))
            for i, (model_name, res) in enumerate(results.items()):
                with cols[i]:
                    is_best = model_name == best_model
                    card_class = "prediction-card best-model" if is_best else "prediction-card"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4>{model_name}{" ★" if is_best else ""}</h4>
                        <p><b>Annual Rate:</b> {res['annual']:.2f} tons/ha/year</p>
                        <p><b>{months}-month Rate:</b> {res['period']:.2f} tons/ha</p>
                        <p><b>Total for {area} ha:</b> {res['total']:.2f} tons</p>
                        <p><small>R²: {res['R2']:.2f}, RMSE: {res['RMSE']:.3f}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show model comparison chart
            st.subheader("📈 Model Comparison")
            
            fig = go.Figure()
            
            for model_name, res in results.items():
                fig.add_trace(go.Bar(
                    x=[model_name],
                    y=[res['annual']],
                    name=model_name,
                    marker_color=res['color'],
                    hovertemplate=f"<b>{model_name}</b><br>Annual Erosion: {res['annual']:.2f} tons/ha/year<br>R²: {res['R2']:.2f}"
                ))
            
            fig.update_layout(
                title="Predicted Annual Erosion Rate by Model",
                xaxis_title="Model",
                yaxis_title="Erosion Rate (tons/ha/year)",
                showlegend=False,
                hovermode="x"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature importance if available
            try:
                if model_choice == "All Models":
                    model_for_importance = best_model
                else:
                    model_for_importance = model_choice
                
                if hasattr(models[model_for_importance], 'feature_importances_'):
                    st.subheader("🔍 Feature Importance")
                    
                    importances = models[model_for_importance].feature_importances_
                    features_list = [info['name'] for info in feature_info.values()]
                    
                    importance_df = pd.DataFrame({
                        'Feature': features_list,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f'Feature Importance ({model_for_importance})',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display feature importance: {str(e)}")
            
            # Show model explanation
            with st.expander("ℹ️ Why this model is recommended", expanded=True):
                st.markdown(model_explanations.get(best_model, "No explanation available."))
        else:
            st.warning("No models selected for prediction.")

else:  # About page
    st.title("📚 About This Tool")
    
    st.markdown("""
    ## Soil Erosion Visualization & Prediction Tool
    
    This interactive tool helps researchers and land managers:
    
    - **Visualize** soil erosion patterns and relationships with environmental factors
    - **Predict** potential erosion rates based on site conditions
    - **Compare** different machine learning models for erosion prediction
    
    ### How It Works
    
    1. **Visualizations**: Explore various charts and graphs showing erosion patterns and model performance
    2. **Prediction**: Input site parameters to get erosion rate estimates from multiple models
    3. **Analysis**: Compare model results and understand which factors most influence erosion
    
    ### Models Included
    
    - **Random Forest**: Ensemble of decision trees, robust and accurate
    - **XGBoost**: Optimized gradient boosting, good for structured data
    - **Gradient Boosting**: Another boosting algorithm, often performs well
    - **Neural Network**: Deep learning approach, less interpretable but powerful
    
    ### Data Sources
    
    The models were trained on data from:
    - Field measurements
    - Remote sensing data
    - Climate records
    - Soil surveys
    
    ### Development Team
    
    This tool was developed by environmental scientists and data scientists to support sustainable land management.
    """)
    
    st.markdown("---")
    st.markdown("© 2023 Soil Erosion Research Group | [Contact Us](mailto:info@soilerosion.org)")