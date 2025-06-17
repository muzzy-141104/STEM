import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_geographic_bias(df):
    """Analyze geographic distribution bias"""
    plt.figure(figsize=(15, 10))
    
    # Create a scatter plot of locations
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                         c=df['erosion_rate'], cmap='YlOrRd',
                         alpha=0.5, s=10)
    plt.colorbar(scatter, label='Erosion Rate')
    plt.title('Geographic Distribution of Samples')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add region boundaries
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
    
    for lat, lon in region_centers:
        plt.plot(lon, lat, 'k+', markersize=10)
    
    plt.savefig('geographic_bias.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate sample density by region
    region_density = {}
    for lat, lon in region_centers:
        mask = ((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2) < 100
        region_density[f"({lat}, {lon})"] = mask.sum()
    
    return region_density

def analyze_land_use_bias(df):
    """Analyze land use distribution bias"""
    plt.figure(figsize=(12, 6))
    
    # Plot land use distribution
    land_use_counts = df['land_use'].value_counts()
    land_use_labels = ['Forest', 'Agriculture', 'Grassland', 'Urban', 'Bare']
    
    plt.bar(land_use_labels, land_use_counts)
    plt.title('Land Use Distribution')
    plt.xlabel('Land Use Type')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    # Add percentage labels
    total = len(df)
    for i, count in enumerate(land_use_counts):
        plt.text(i, count, f'{count/total*100:.1f}%', 
                ha='center', va='bottom')
    
    plt.savefig('land_use_bias.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze erosion rates by land use
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='land_use', y='erosion_rate', data=df)
    plt.title('Erosion Rate Distribution by Land Use')
    plt.xlabel('Land Use Type')
    plt.ylabel('Erosion Rate')
    plt.xticks(range(5), land_use_labels, rotation=45)
    plt.savefig('erosion_by_landuse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return land_use_counts

def analyze_climate_bias(df):
    """Analyze climate data bias"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Temperature distribution
    sns.histplot(data=df, x='temperature_c', ax=axes[0,0])
    axes[0,0].set_title('Temperature Distribution')
    
    # Humidity distribution
    sns.histplot(data=df, x='humidity_percent', ax=axes[0,1])
    axes[0,1].set_title('Humidity Distribution')
    
    # Rainfall erosivity distribution
    sns.histplot(data=df, x='rainfall_erosivity', ax=axes[1,0])
    axes[1,0].set_title('Rainfall Erosivity Distribution')
    
    # Temperature vs Elevation
    sns.scatterplot(data=df, x='elevation_m', y='temperature_c', ax=axes[1,1])
    axes[1,1].set_title('Temperature vs Elevation')
    
    plt.tight_layout()
    plt.savefig('climate_bias.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate climate correlations
    climate_corr = df[['temperature_c', 'humidity_percent', 'rainfall_erosivity', 'elevation_m']].corr()
    return climate_corr

def analyze_feature_correlations(df):
    """Analyze feature correlation biases"""
    # Select relevant features
    features = ['slope_angle_deg', 'NDVI', 'rainfall_erosivity', 'soil_moisture_percent',
                'elevation_m', 'slope_length', 'aspect_deg', 'plan_curvature',
                'profile_curvature', 'soil_organic_matter', 'soil_depth', 'erosion_rate']
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

def analyze_soil_property_bias(df):
    """Analyze soil property biases"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Organic matter distribution
    sns.histplot(data=df, x='soil_organic_matter', ax=axes[0,0])
    axes[0,0].set_title('Soil Organic Matter Distribution')
    
    # Soil depth distribution
    sns.histplot(data=df, x='soil_depth', ax=axes[0,1])
    axes[0,1].set_title('Soil Depth Distribution')
    
    # Soil texture distribution
    soil_texture_counts = df['soil_texture'].value_counts()
    axes[1,0].bar(soil_texture_counts.index, soil_texture_counts.values)
    axes[1,0].set_title('Soil Texture Distribution')
    plt.xticks(rotation=45)
    
    # Erosion rate by soil texture
    sns.boxplot(data=df, x='soil_texture', y='erosion_rate', ax=axes[1,1])
    axes[1,1].set_title('Erosion Rate by Soil Texture')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('soil_property_bias.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'organic_matter_stats': df['soil_organic_matter'].describe(),
        'soil_depth_stats': df['soil_depth'].describe(),
        'texture_distribution': soil_texture_counts
    }

def analyze_temporal_bias(df):
    """Analyze temporal biases"""
    plt.figure(figsize=(12, 6))
    
    # Plot seasonal distribution
    season_counts = df['season'].value_counts()
    plt.bar(season_counts.index, season_counts.values)
    plt.title('Seasonal Distribution')
    plt.xlabel('Season')
    plt.ylabel('Number of Samples')
    
    # Add percentage labels
    total = len(df)
    for i, count in enumerate(season_counts):
        plt.text(i, count, f'{count/total*100:.1f}%', 
                ha='center', va='bottom')
    
    plt.savefig('temporal_bias.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze erosion rates by season
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='season', y='erosion_rate')
    plt.title('Erosion Rate Distribution by Season')
    plt.xlabel('Season')
    plt.ylabel('Erosion Rate')
    plt.savefig('erosion_by_season.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return season_counts

def main():
    print("Loading dataset...")
    df = pd.read_csv("large_soil_erosion_dataset.csv")
    
    print("\nAnalyzing geographic bias...")
    region_density = analyze_geographic_bias(df)
    print("Geographic bias analysis saved as 'geographic_bias.png'")
    
    print("\nAnalyzing land use bias...")
    land_use_distribution = analyze_land_use_bias(df)
    print("Land use bias analysis saved as 'land_use_bias.png' and 'erosion_by_landuse.png'")
    
    print("\nAnalyzing climate bias...")
    climate_correlations = analyze_climate_bias(df)
    print("Climate bias analysis saved as 'climate_bias.png'")
    
    print("\nAnalyzing feature correlations...")
    feature_correlations = analyze_feature_correlations(df)
    print("Feature correlation analysis saved as 'feature_correlations.png'")
    
    print("\nAnalyzing soil property bias...")
    soil_property_stats = analyze_soil_property_bias(df)
    print("Soil property bias analysis saved as 'soil_property_bias.png'")
    
    print("\nAnalyzing temporal bias...")
    seasonal_distribution = analyze_temporal_bias(df)
    print("Temporal bias analysis saved as 'temporal_bias.png' and 'erosion_by_season.png'")
    
    # Print summary statistics
    print("\nBias Analysis Summary:")
    print("\n1. Geographic Distribution:")
    for region, density in region_density.items():
        print(f"  {region}: {density} samples")
    
    print("\n2. Land Use Distribution:")
    for land_use, count in land_use_distribution.items():
        print(f"  {land_use}: {count} samples ({count/len(df)*100:.1f}%)")
    
    print("\n3. Climate Correlations:")
    print(climate_correlations)
    
    print("\n4. Soil Property Statistics:")
    print("\nOrganic Matter:")
    print(soil_property_stats['organic_matter_stats'])
    print("\nSoil Depth:")
    print(soil_property_stats['soil_depth_stats'])
    print("\nSoil Texture Distribution:")
    print(soil_property_stats['texture_distribution'])
    
    print("\n5. Seasonal Distribution:")
    for season, count in seasonal_distribution.items():
        print(f"  {season}: {count} samples ({count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main() 