import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def comprehensive_model_evaluation(model, X, y, X_test, y_test, y_pred):
    """
    Comprehensive evaluation of the soil erosion prediction model
    """
    print("📊 COMPREHENSIVE MODEL EVALUATION REPORT")
    print("=" * 60)
    
    # 1. Basic Performance Metrics
    print("\n1️⃣ BASIC PERFORMANCE METRICS")
    print("-" * 40)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    nse = nash_sutcliffe_efficiency(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    
    print(f"Root Mean Square Error (RMSE): {rmse:.4f} kg/m²")
    print(f"Mean Absolute Error (MAE): {mae:.4f} kg/m²")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Bias: {bias:.4f} kg/m²")
    
    # 2. Cross-Validation Performance
    print("\n2️⃣ CROSS-VALIDATION PERFORMANCE (5-Fold)")
    print("-" * 40)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
    
    print(f"CV R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    
    # 3. Model Classification
    print("\n3️⃣ MODEL PERFORMANCE CLASSIFICATION")
    print("-" * 40)
    
    performance_rating = classify_model_performance(r2, nse, mape)
    print(f"Overall Model Rating: {performance_rating}")
    
    # 4. Residual Analysis
    print("\n4️⃣ RESIDUAL ANALYSIS")
    print("-" * 40)
    
    residuals = y_test - y_pred
    residual_stats = analyze_residuals(residuals)
    
    for key, value in residual_stats.items():
        print(f"{key}: {value}")
    
    # 5. Feature Importance (if available)
    print("\n5️⃣ FEATURE IMPORTANCE")
    print("-" * 40)
    
    try:
        feature_importance = get_feature_importance(model, X)
        print("Top 5 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance.head().items(), 1):
            print(f"{i}. {feature}: {importance:.4f}")
    except Exception as e:
        print(f"Feature importance analysis not available: {e}")
    
    return {
        'rmse': rmse, 'mae': mae, 'r2': r2, 'nse': nse, 'mape': mape, 'bias': bias,
        'cv_r2_mean': cv_scores.mean(), 'cv_r2_std': cv_scores.std(),
        'cv_rmse_mean': cv_rmse.mean(), 'cv_rmse_std': cv_rmse.std(),
        'performance_rating': performance_rating,
        'residual_stats': residual_stats
    }

def nash_sutcliffe_efficiency(y_true, y_pred):
    """Calculate Nash-Sutcliffe Efficiency"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mean_observed = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - mean_observed) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else np.nan

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def classify_model_performance(r2, nse, mape):
    """Classify model performance based on multiple metrics"""
    if r2 >= 0.85 and nse >= 0.8 and mape <= 15:
        return "🟢 EXCELLENT"
    elif r2 >= 0.70 and nse >= 0.65 and mape <= 25:
        return "🟡 GOOD"
    elif r2 >= 0.50 and nse >= 0.40 and mape <= 40:
        return "🟠 ACCEPTABLE"
    else:
        return "🔴 NEEDS IMPROVEMENT"

def analyze_residuals(residuals):
    """Analyze residual statistics"""
    return {
        'Mean Residual': f"{np.mean(residuals):.4f}",
        'Std Residual': f"{np.std(residuals):.4f}",
        'Skewness': f"{stats.skew(residuals):.4f}",
        'Kurtosis': f"{stats.kurtosis(residuals):.4f}",
        'Normality Test (p-value)': f"{stats.shapiro(residuals)[1]:.4f}"
    }

def get_feature_importance(model, X):
    """Extract feature importance from the model"""
    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']
    
    # Transform a sample to get feature names
    X_sample = X.head(1)
    X_transformed = preprocessor.transform(X_sample)
    
    # Get feature names
    numeric_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out(['land_use'])
    feature_names = list(numeric_features) + list(cat_features)
    
    # Get importance scores
    importance_scores = regressor.feature_importances_
    
    # Create Series and sort
    importance_df = pd.Series(importance_scores, index=feature_names)
    return importance_df.sort_values(ascending=False)

def create_detailed_plots(y_test, y_pred):
    """Create comprehensive visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
    
    # Plot 1: Predicted vs Observed
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Observed Erosion (kg/m²)')
    axes[0, 0].set_ylabel('Predicted Erosion (kg/m²)')
    axes[0, 0].set_title('Predicted vs Observed')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² to the plot
    r2 = r2_score(y_test, y_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Residuals vs Predicted
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Erosion (kg/m²)')
    axes[0, 1].set_ylabel('Residuals (kg/m²)')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram of Residuals
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals (kg/m²)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Q-Q Plot for Residuals
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional performance summary plot
    create_performance_summary(y_test, y_pred)

def create_performance_summary(y_test, y_pred):
    """Create a performance summary visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'NSE': nash_sutcliffe_efficiency(y_test, y_pred),
        'MAPE (%)': mean_absolute_percentage_error(y_test, y_pred)
    }
    
    # Create bar plot
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'gold', 'coral', 'plum'])
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Model Performance Metrics Summary', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
