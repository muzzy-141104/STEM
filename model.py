# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# # Function to calculate NSE (Nash-Sutcliffe Efficiency)
# def nash_sutcliffe_efficiency(y_true, y_pred):
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     mean_observed = np.mean(y_true)
#     numerator = np.sum((y_true - y_pred) ** 2)
#     denominator = np.sum((y_true - mean_observed) ** 2)
#     return 1 - (numerator / denominator) if denominator != 0 else np.nan

# # Step 1: Load and preprocess data
# def preprocess_data(file_path):
#     # Load CSV
#     df = pd.read_csv(file_path)

#     # Generate synthetic erosion rate (kg/m²) based on paper's key drivers
#     df['erosion_rate'] = (
#         0.5 * (df['slope_angle_deg'] / 28.3) +  # Normalize slope (max 28.3°)
#         0.1 * ((df['plan_curvature'] + 0.56) / 1.54) +  # Normalize curvature
#         -0.3 * ((df['NDVI'] + 1) / 2) +  # NDVI (-1 to 1) to vegetation proxy (0–1)
#         0.2 * (df['rainfall_erosivity'] / df['rainfall_erosivity'].max()) +  # Normalize
#         0.05 * (df['soil_moisture_percent'] / 100)  # Minor effect
#     ) * 10  # Scale to realistic range (0–10 kg/m²)
#     df['erosion_rate'] = df['erosion_rate'].clip(0)  # Ensure non-negative

#     # Features and target
#     features = ['soil_moisture_percent', 'temperature_c', 'humidity_percent',
#                 'B02', 'B03', 'B04', 'B08', 'B11', 'NDVI', 'BSI', 'NDWI',
#                 'elevation_m', 'slope_angle_deg', 'land_use', 'plan_curvature',
#                 'rainfall_erosivity']
#     target = 'erosion_rate'

#     # Define preprocessing pipeline
#     numeric_features = [f for f in features if f != 'land_use']
#     categorical_features = ['land_use']
    
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', MinMaxScaler(), numeric_features),
#             ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
#         ])

#     # Apply preprocessing
#     X = df[features]
#     y = df[target]
    
#     return X, y, preprocessor

# # Step 2: Train BRT model
# def train_brt_model(X, y, preprocessor):
#     # Split data (75% training, 25% testing)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#     # Create pipeline
#     model = Pipeline([
#         ('preprocessor', preprocessor),
#         ('regressor', xgb.XGBRegressor(
#             objective='reg:squarederror',
#             n_estimators=100,
#             learning_rate=0.1,
#             max_depth=5,
#             random_state=42
#         ))
#     ])

#     # Train model
#     model.fit(X_train, y_train)

#     # Predict on test set
#     y_pred = model.predict(X_test)

#     # Evaluate model
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     nrmse = rmse / (y_test.max() - y_test.min()) if (y_test.max() - y_test.min()) != 0 else np.nan
#     r2 = r2_score(y_test, y_pred)
#     nse = nash_sutcliffe_efficiency(y_test, y_pred)

#     print(f"Test Set Performance:")
#     print(f"RMSE: {rmse:.2f} kg/m²")
#     print(f"NRMSE: {nrmse:.2f}")
#     print(f"R-squared: {r2:.2f}")
#     print(f"NSE: {nse:.2f}")

#     return model, X_test, y_test, y_pred

# # Step 3: Visualize results
# def plot_results(y_test, y_pred):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Observed')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='1:1 Line')
#     plt.xlabel('Observed Soil Erosion (kg/m²)')
#     plt.ylabel('Predicted Soil Erosion (kg/m²)')
#     plt.title('BRT Model: Predicted vs Observed Soil Erosion')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('soil_erosion_csv_prediction.png')
#     plt.close()

# # Main execution
# if __name__ == "__main__":
#     # Replace with your CSV file path
#     file_path = "soil_data.csv"  # Update this path

#     # Preprocess data
#     X, y, preprocessor = preprocess_data("soil_erosion_dataset_with_rate.csv")

#     # Train and evaluate model
#     model, X_test, y_test, y_pred = train_brt_model(X, y, preprocessor)

#     # Basic visualization
#     plot_results(y_test, y_pred)

#     # Comprehensive evaluation
#     print("\n" + "="*60)
#     print("🚀 RUNNING COMPREHENSIVE MODEL EVALUATION...")
#     print("="*60)
    
#     # Import and run comprehensive evaluation
#     try:
#         from model_evaluation import comprehensive_model_evaluation, create_detailed_plots
#         metrics = comprehensive_model_evaluation(model, X, y, X_test, y_test, y_pred)
#         create_detailed_plots(y_test, y_pred)
#         print(f"\n📊 Detailed evaluation plots saved as:")
#         print(f"   - 'comprehensive_model_evaluation.png'")
#         print(f"   - 'model_performance_summary.png'")
        
#         # Print final recommendation
#         print(f"\n🎯 FINAL RECOMMENDATION:")
#         print(f"   Model Performance: {metrics['performance_rating']}")
#         if 'EXCELLENT' in metrics['performance_rating']:
#             print(f"   ✅ Your model is performing excellently! Ready for deployment.")
#         elif 'GOOD' in metrics['performance_rating']:
#             print(f"   ✅ Your model shows good performance. Consider fine-tuning for better results.")
#         elif 'ACCEPTABLE' in metrics['performance_rating']:
#             print(f"   ⚠️  Model performance is acceptable but has room for improvement.")
#         else:
#             print(f"   ❌ Model needs significant improvement before deployment.")
            
#     except ImportError as e:
#         print(f"❌ Error importing evaluation module: {e}")
#         print("Note: Make sure model_evaluation.py is in the same directory")

#     # Save model for Simulink integration
#     try:
#         model.named_steps['regressor'].save_model('brt_csv_model.json')
#         print(f"\n💾 Model saved as 'brt_csv_model.json' for Simulink integration")
#     except Exception as e:
#         print(f"❌ Error saving model: {e}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate_models():
    print("\n🌱 Loading the generated dataset...")
    df = pd.read_csv("large_soil_erosion_dataset.csv")
    print(f"Dataset loaded with {len(df)} samples and {len(df.columns)} features")
    
    # Prepare features and target
    print("\n🔧 Preparing features...")
    features = ['slope_angle_deg', 'NDVI', 'rainfall_erosivity', 'soil_moisture_percent',
                'elevation_m', 'slope_length', 'aspect_deg', 'plan_curvature',
                'profile_curvature', 'soil_organic_matter', 'soil_depth']
    
    X = df[features]
    y = df['erosion_rate']
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler
    joblib.dump(scaler, 'soil_erosion_scaler.joblib')
    print("Scaler saved as 'soil_erosion_scaler.joblib'")
    
    # Split data
    print("\n📊 Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize models with optimized parameters
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'Neural Network': MLPRegressor(
            hidden_layer_sizes=(200, 100),
            max_iter=2000,
            learning_rate_init=0.001,
            random_state=42
        )
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        print(f"✅ {name} training completed")
        
        # Save model
        joblib.dump(model, f'soil_erosion_model_{name.lower().replace(" ", "_")}.joblib')
        print(f"💾 Model saved as 'soil_erosion_model_{name.lower().replace(" ", "_")}.joblib'")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        print(f"📊 Performing cross-validation for {name}...")
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
        
        # Create prediction plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Erosion Rate')
        plt.ylabel('Predicted Erosion Rate')
        plt.title(f'Actual vs Predicted Erosion Rates - {name}')
        plt.tight_layout()
        plt.savefig(f'prediction_accuracy_{name.lower().replace(" ", "_")}.png')
        plt.close()
        
        # Create feature importance plot if available
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.title(f'Feature Importance - {name}')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.lower().replace(" ", "_")}.png')
            plt.close()
        
        print(f"📈 {name} Performance Metrics:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Cross-validated R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
    
    # Create comparison plots
    print("\n📊 Creating comparison plots...")
    
    # Model performance comparison
    plt.figure(figsize=(12, 6))
    r2_scores = [results[name]['R2'] for name in models.keys()]
    cv_r2_scores = [results[name]['CV_R2_mean'] for name in models.keys()]
    x = np.arange(len(models.keys()))
    width = 0.35
    plt.bar(x - width/2, r2_scores, width, label='R² Score')
    plt.bar(x + width/2, cv_r2_scores, width, label='Cross-validated R²')
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()
    
    # Error metrics comparison
    plt.figure(figsize=(12, 6))
    rmse_scores = [results[name]['RMSE'] for name in models.keys()]
    mae_scores = [results[name]['MAE'] for name in models.keys()]
    plt.bar(x - width/2, rmse_scores, width, label='RMSE')
    plt.bar(x + width/2, mae_scores, width, label='MAE')
    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title('Model Error Metrics')
    plt.xticks(x, models.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_errors.png')
    plt.close()
    
    print("\n✨ Model training and evaluation completed!")
    print("\n📊 Final Results Summary:")
    for name in models.keys():
        print(f"\n{name}:")
        print(f"  R² Score: {results[name]['R2']:.4f}")
        print(f"  Cross-validated R²: {results[name]['CV_R2_mean']:.4f} (±{results[name]['CV_R2_std']:.4f})")
        print(f"  RMSE: {results[name]['RMSE']:.4f}")
        print(f"  MAE: {results[name]['MAE']:.4f}")

if __name__ == "__main__":
    print("🌱 SOIL EROSION MODEL OPTIMIZATION PIPELINE")
    print("="*60)
    train_and_evaluate_models()