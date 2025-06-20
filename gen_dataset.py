import numpy as np
import pandas as pd
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
import shap
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points - increased for better model training
n_samples = 50000  # Increased back to 50000 for better accuracy and visualization

def generate_correlated_climate_data(n_samples):
    """Generate realistic climate data with proper correlations"""
    
    # Generate elevation first (affects temperature and humidity)
    elevation = np.random.exponential(scale=300, size=n_samples)
    elevation = np.clip(elevation, 0, 4000)
    
    # Temperature decreases with elevation (lapse rate: ~6.5°C per 1000m)
    base_temp = np.random.normal(25, 8, n_samples)  # More realistic global range
    temperature = base_temp - (elevation * 6.5 / 1000)
    temperature = np.clip(temperature, -10, 45)  # Realistic bounds
    
    # Humidity correlates with temperature and location
    base_humidity = 50 + 30 * np.random.beta(2, 2, n_samples)  # Beta distribution for realism
    # Higher humidity in warmer areas (but not linearly)
    temp_humidity_factor = 1 + 0.3 * np.tanh((temperature - 20) / 10)
    humidity = base_humidity * temp_humidity_factor
    humidity = np.clip(humidity, 20, 95)
    
    # Soil moisture depends on rainfall, temperature, and land use
    rainfall = np.random.gamma(shape=2, scale=1000, size=n_samples)  # More realistic rainfall distribution
    soil_moisture = np.clip(
        30 + 0.3 * rainfall/1000 - 0.2 * (temperature - 20) + np.random.normal(0, 5, n_samples),
        0, 100
    )
    
    return elevation, temperature, humidity, soil_moisture, rainfall

def generate_realistic_geography(n_samples):
    """Generate geographically consistent locations"""
    
    # Create clusters around realistic agricultural/erosion-prone regions
    region_centers = [
        (40, -100),   # US Great Plains
        (45, 2),      # European farmlands
        (-15, -50),   # South American croplands
        (20, 77),     # Indian subcontinent
        (-25, 135),   # Australian agricultural areas
        (35, 105),    # Chinese agricultural regions
        (30, -90),    # Mississippi River Basin
        (-5, 120),    # Indonesian agricultural areas
        (15, -90),    # Central American agricultural regions
        (50, 30),     # Ukrainian agricultural areas
        (35, 139),    # Japanese agricultural areas
        (28, 77),     # Indian agricultural areas
        (-33, 151),   # Australian coastal areas
        (37, -122),   # California agricultural areas
        (52, 13)      # German agricultural areas
    ]
    
    latitudes = []
    longitudes = []
    
    for _ in range(n_samples):
        # Choose a region
        center_lat, center_lon = region_centers[np.random.choice(len(region_centers))]
        
        # Add noise around the center (realistic spread)
        lat_noise = np.random.normal(0, 8)  # ~8 degree spread
        lon_noise = np.random.normal(0, 12)  # ~12 degree spread
        
        lat = np.clip(center_lat + lat_noise, -60, 70)  # Avoid extreme poles
        lon = center_lon + lon_noise
        if lon > 180:
            lon -= 360
        elif lon < -180:
            lon += 360
            
        latitudes.append(lat)
        longitudes.append(lon)
    
    return np.array(latitudes), np.array(longitudes)

def generate_sentinel_bands_realistic(n_samples, ndvi_target, land_use):
    """Generate realistic Sentinel-2 bands based on land use and NDVI"""
    
    # Initialize arrays
    B02 = np.zeros(n_samples)  # Blue
    B03 = np.zeros(n_samples)  # Green  
    B04 = np.zeros(n_samples)  # Red
    B08 = np.zeros(n_samples)  # NIR
    B11 = np.zeros(n_samples)  # SWIR
    
    for i in range(n_samples):
        lu = land_use[i]
        target_ndvi = ndvi_target[i]
        
        if lu == 0:  # Forest - high NIR, moderate visible
            B02[i] = np.random.uniform(0.02, 0.08)
            B03[i] = np.random.uniform(0.03, 0.12)
            B04[i] = np.random.uniform(0.02, 0.08)
            B08[i] = np.random.uniform(0.4, 0.8)
            B11[i] = np.random.uniform(0.1, 0.25)
            
        elif lu == 1:  # Agriculture - variable based on crop type/season
            B02[i] = np.random.uniform(0.04, 0.15)
            B03[i] = np.random.uniform(0.05, 0.20)
            B04[i] = np.random.uniform(0.04, 0.15)
            B08[i] = np.random.uniform(0.2, 0.6)
            B11[i] = np.random.uniform(0.15, 0.35)
            
        elif lu == 2:  # Grassland - moderate values
            B02[i] = np.random.uniform(0.05, 0.12)
            B03[i] = np.random.uniform(0.06, 0.18)
            B04[i] = np.random.uniform(0.05, 0.12)
            B08[i] = np.random.uniform(0.25, 0.5)
            B11[i] = np.random.uniform(0.2, 0.4)
            
        elif lu == 3:  # Urban - lower NIR, higher visible
            B02[i] = np.random.uniform(0.08, 0.25)
            B03[i] = np.random.uniform(0.1, 0.3)
            B04[i] = np.random.uniform(0.08, 0.25)
            B08[i] = np.random.uniform(0.15, 0.35)
            B11[i] = np.random.uniform(0.25, 0.45)
            
        else:  # Bare soil - high SWIR, low NIR
            B02[i] = np.random.uniform(0.1, 0.3)
            B03[i] = np.random.uniform(0.12, 0.35)
            B04[i] = np.random.uniform(0.1, 0.3)
            B08[i] = np.random.uniform(0.1, 0.25)
            B11[i] = np.random.uniform(0.3, 0.5)
    
    return B02, B03, B04, B08, B11

def generate_terrain_features(n_samples, elevation):
    """Generate realistic terrain features"""
    
    # Slope angle (degrees) - correlated with elevation
    slope_base = np.random.beta(2, 5, n_samples) * 45  # Most slopes are gentle
    slope = np.clip(slope_base + (elevation/4000) * 15, 0, 60)
    
    # Plan curvature (affects water flow)
    plan_curvature = np.random.normal(0, 0.1, n_samples)
    
    # Profile curvature (affects erosion)
    profile_curvature = np.random.normal(0, 0.15, n_samples)
    
    # Aspect (degrees from north)
    aspect = np.random.uniform(0, 360, n_samples)
    
    return slope, plan_curvature, profile_curvature, aspect

def calculate_erosion_rate(data):
    """Calculate erosion rate based on RUSLE factors"""
    
    # R = Rainfall erosivity factor
    R = data['rainfall_erosivity'] / 10000  # Normalize
    
    # K = Soil erodibility factor (0.1-0.5)
    K = np.random.uniform(0.1, 0.5, len(data))
    
    # LS = Slope length and steepness factor
    slope_rad = np.radians(data['slope_angle_deg'])
    LS = (data['slope_length'] / 22.1) ** 0.5 * (np.sin(slope_rad) / 0.0896) ** 1.3
    
    # C = Cover factor (based on land use and NDVI)
    C = np.where(data['land_use'] == 0, 0.001,  # Forest
                np.where(data['land_use'] == 1, 0.1,  # Agriculture
                        np.where(data['land_use'] == 2, 0.05,  # Grassland
                                np.where(data['land_use'] == 3, 0.001,  # Urban
                                        0.5))))  # Bare soil
    C = C * (1 - (data['NDVI'] + 1) / 2)  # Adjust by vegetation
    
    # P = Practice factor (conservation practices)
    P = np.random.uniform(0.5, 1.0, len(data))
    
    # Calculate erosion rate (tons/ha/year)
    erosion_rate = R * K * LS * C * P
    
    # Add some noise
    erosion_rate = erosion_rate * (1 + np.random.normal(0, 0.1, len(data)))
    erosion_rate = np.clip(erosion_rate, 0, None)
    
    return erosion_rate

def train_and_evaluate_models(df):
    """Train multiple models and evaluate their performance"""
    
    print("\nPreparing data for model training...")
    # Prepare features and target
    features = ['slope_angle_deg', 'NDVI', 'rainfall_erosivity', 'soil_moisture_percent',
                'elevation_m', 'slope_length', 'aspect_deg', 'plan_curvature',
                'profile_curvature', 'soil_organic_matter', 'soil_depth']
    
    X = df[features]
    y = df['erosion_rate']
    
    print("Scaling features...")
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Splitting data into train and test sets...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize models with optimized parameters for better accuracy
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,  # Increased for better accuracy
            max_depth=15,      # Increased for better accuracy
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all available cores
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,  # Increased for better accuracy
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,  # Increased for better accuracy
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(200, 100),  # Increased network size
            max_iter=2000,                  # Increased iterations
            learning_rate_init=0.001,
            random_state=42
        )
    }
    
    # Store results
    results = {}
    feature_importance = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        print(f"Model parameters: {model.get_params()}")
        
        # Train model
        model.fit(X_train, y_train)
        print(f"Completed training {name}")
        
        # Save model
        joblib.dump(model, f'soil_erosion_model_{name.lower().replace(" ", "_")}.joblib')
        print(f"Saved {name} model")
        
        # Make predictions
        print(f"Making predictions with {name}...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        print(f"Calculating performance metrics for {name}...")
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Calculating cross-validation scores for {name}...")
        # Cross-validation with parallel processing
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2', n_jobs=-1)
        
        # Store results
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV_R2_mean': cv_scores.mean(),
            'CV_R2_std': cv_scores.std()
        }
        
        # Store feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = model.feature_importances_
        elif name == 'Neural Network':
            feature_importance[name] = np.abs(model.coefs_[0]).mean(axis=1)
        
        print(f"Completed evaluation for {name}")
        print(f"R² Score: {r2:.4f}")
        print(f"Cross-validated R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
    
    # Create visualizations
    create_performance_visualizations(results)
    create_feature_importance_visualizations(feature_importance, features)
    create_prediction_visualizations(models, X_test, y_test, features)
    create_feature_correlation_visualization(df[features + ['erosion_rate']])
    
    # After model training and evaluation, add advanced visualizations for Random Forest
    print("\nGenerating advanced visualizations for Random Forest model...")
    rf_model = models['Random Forest']
    create_advanced_visualizations(rf_model, X_scaled, y, features, df)
    
    # Print results
    print("\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  R² Score: {metrics['R2']:.4f}")
        print(f"  Cross-validated R²: {metrics['CV_R2_mean']:.4f} (±{metrics['CV_R2_std']:.4f})")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
    
    print("\nVisualization files saved:")
    print("- model_performance.png: Comparison of model performance")
    print("- model_errors.png: Comparison of model error metrics")
    print("- feature_importance_*.png: Feature importance for each model")
    print("- prediction_accuracy_*.png: Prediction accuracy for each model")
    print("- feature_correlation.png: Correlation matrix of features")
    print("\nModel files saved:")
    print("- soil_erosion_scaler.joblib: Feature scaler")
    print("- soil_erosion_model_*.joblib: Trained models")

def create_performance_visualizations(results):
    """Create visualizations of model performance"""
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    
    # Plot performance metrics
    plt.figure(figsize=(12, 6))
    results_df[['R2', 'CV_R2_mean']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('public/model_performance.png')
    plt.close()
    
    # Plot error metrics
    plt.figure(figsize=(12, 6))
    results_df[['RMSE', 'MAE']].plot(kind='bar')
    plt.title('Model Error Metrics')
    plt.ylabel('Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('public/model_errors.png')
    plt.close()

def create_feature_importance_visualizations(feature_importance, features):
    """Create visualizations of feature importance"""
    
    # Plot feature importance for each model
    for model_name, importance in feature_importance.items():
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'public/feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

def create_prediction_visualizations(models, X_test, y_test, features):
    """Create visualizations of predictions"""
    
    # Plot actual vs predicted for each model
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Erosion Rate')
        plt.ylabel('Predicted Erosion Rate')
        plt.title(f'Actual vs Predicted Erosion Rates - {name}')
        plt.tight_layout()
        plt.savefig(f'public/prediction_accuracy_{name.lower().replace(" ", "_")}.png')
        plt.close()

def create_feature_correlation_visualization(df):
    """Create correlation matrix visualization"""
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('public/feature_correlation.png')
    plt.close()

def create_advanced_visualizations(model, X, y, features, df):
    print("\nGenerating advanced visualizations...")
    
    # 1. Enhanced Scatter Plot Matrix for Top Features
    print("Generating enhanced scatter plot matrix...")
    importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(len(features))
    top_idx = np.argsort(importances)[-4:][::-1]  # Get top 4 features
    top_features = [features[i] for i in top_idx]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, feature in enumerate(top_features):
        # Use hexbin for better visualization of dense data
        hb = axes[idx].hexbin(df[feature], df['erosion_rate'], 
                             gridsize=50, cmap='YlOrRd')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Erosion Rate')
        axes[idx].set_title(f'{feature} vs Erosion Rate')
        plt.colorbar(hb, ax=axes[idx], label='Count')
    
    plt.tight_layout()
    plt.savefig('public/top_features_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Enhanced scatter plot matrix saved as 'top_features_scatter.png'")

    # 2. Enhanced Distribution Plots
    print("Generating enhanced distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, feature in enumerate(top_features):
        # Create KDE plot with rug plot
        sns.kdeplot(data=df, x=feature, ax=axes[idx], fill=True)
        sns.rugplot(data=df, x=feature, ax=axes[idx], alpha=0.1)
        axes[idx].set_title(f'Distribution of {feature}')
    
    plt.tight_layout()
    plt.savefig('public/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Distribution plots saved as 'feature_distributions.png'")

    # 3. Enhanced Correlation Heatmap
    print("Generating enhanced correlation heatmap...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[top_features + ['erosion_rate']].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                fmt='.2f', square=True, linewidths=.5)
    plt.title('Correlation Matrix of Top Features')
    plt.tight_layout()
    plt.savefig('public/enhanced_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Enhanced correlation heatmap saved as 'enhanced_correlation.png'")

    # 4. Enhanced Box Plots by Land Use
    print("Generating enhanced box plots...")
    plt.figure(figsize=(15, 8))
    land_use_labels = ['Forest', 'Agriculture', 'Grassland', 'Urban', 'Bare']
    df['land_use_label'] = df['land_use'].map(dict(enumerate(land_use_labels)))
    
    sns.boxplot(data=df, x='land_use_label', y='erosion_rate', 
                palette='Set3', showfliers=False)
    plt.xticks(rotation=45)
    plt.title('Erosion Rate Distribution by Land Use')
    plt.tight_layout()
    plt.savefig('public/land_use_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Enhanced box plots saved as 'land_use_boxplot.png'")

    # 5. Enhanced Geographic Distribution
    print("Generating enhanced geographic distribution...")
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                         c=df['erosion_rate'], cmap='YlOrRd',
                         alpha=0.5, s=10)
    plt.colorbar(scatter, label='Erosion Rate')
    plt.title('Geographic Distribution of Erosion Rates')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig('public/geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Enhanced geographic distribution saved as 'geographic_distribution.png'")

    # 6. Enhanced Feature Importance Plot
    print("Generating enhanced feature importance plot...")
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('public/enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Enhanced feature importance plot saved as 'enhanced_feature_importance.png'")

    # 7. Enhanced Residual Plot
    print("Generating enhanced residual plot...")
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    plt.figure(figsize=(12, 8))
    plt.scatter(y_pred, residuals, alpha=0.1, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Erosion Rate')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig('public/enhanced_residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Enhanced residual plot saved as 'enhanced_residual_plot.png'")

    print("\nAll enhanced visualizations have been generated successfully!")

def main():
    print("Generating synthetic soil erosion dataset with 50,000 samples...")
    
    # Generate base data
    elevation, temperature, humidity, soil_moisture, rainfall = generate_correlated_climate_data(n_samples)
    latitude, longitude = generate_realistic_geography(n_samples)
    
    # Generate land use with realistic distribution
    land_use = np.random.choice([0, 1, 2, 3, 4], n_samples, 
                              p=[0.3, 0.3, 0.2, 0.1, 0.1])  # Forest, Agriculture, Grassland, Urban, Bare
    
    # Generate terrain features
    slope, plan_curvature, profile_curvature, aspect = generate_terrain_features(n_samples, elevation)
    
    # Generate slope length (meters)
    slope_length = np.random.exponential(scale=100, size=n_samples)
    slope_length = np.clip(slope_length, 10, 1000)
    
    # Generate initial NDVI targets based on land use
    ndvi_target = np.where(land_use == 0, np.random.uniform(0.6, 0.9, n_samples),  # Forest
                          np.where(land_use == 1, np.random.uniform(0.3, 0.8, n_samples),  # Agriculture
                                  np.where(land_use == 2, np.random.uniform(0.4, 0.7, n_samples),  # Grassland
                                          np.where(land_use == 3, np.random.uniform(0.1, 0.3, n_samples),  # Urban
                                                  np.random.uniform(-0.1, 0.2, n_samples)))))  # Bare soil
    
    # Generate Sentinel-2 bands
    B02, B03, B04, B08, B11 = generate_sentinel_bands_realistic(n_samples, ndvi_target, land_use)
    
    # Calculate indices
    NDVI = (B08 - B04) / (B08 + B04 + 1e-6)
    BSI = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02) + 1e-6)
    NDWI = (B03 - B11) / (B03 + B11 + 1e-6)
    
    # Create DataFrame
    data = {
        'latitude': latitude,
        'longitude': longitude,
        'elevation_m': elevation,
        'slope_angle_deg': slope,
        'slope_length': slope_length,
        'aspect_deg': aspect,
        'plan_curvature': plan_curvature,
        'profile_curvature': profile_curvature,
        'soil_moisture_percent': soil_moisture,
        'temperature_c': temperature,
        'humidity_percent': humidity,
        'rainfall_erosivity': rainfall,
        'B02': B02,
        'B03': B03,
        'B04': B04,
        'B08': B08,
        'B11': B11,
        'NDVI': NDVI,
        'BSI': BSI,
        'NDWI': NDWI,
        'land_use': land_use
    }
    
    df = pd.DataFrame(data)
    
    # Calculate erosion rate
    df['erosion_rate'] = calculate_erosion_rate(df)
    
    # Add soil properties
    df['soil_organic_matter'] = np.random.uniform(0.5, 5.0, n_samples)  # %
    df['soil_texture'] = np.random.choice(['Sandy', 'Loamy', 'Clayey'], n_samples)
    df['soil_depth'] = np.random.uniform(0.5, 2.0, n_samples)  # meters
    
    # Add seasonal information
    df['season'] = np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples)
    
    # Save to CSV
    df.to_csv("large_soil_erosion_dataset.csv", index=False)
    print(f"Large dataset saved with {len(df)} rows and {len(df.columns)} columns.")
    
    # Train and evaluate models with optimized parameters
    print("\nTraining and evaluating multiple models...")
    
    train_and_evaluate_models(df)

if __name__ == "__main__":
    main()