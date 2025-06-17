import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def generate_balanced_geography(n_samples):
    """Generate more balanced geographic distribution"""
    # Define more diverse regions with balanced samples
    regions = [
        (40, -100), (45, 2), (-15, -50), (20, 77), (-25, 135),
        (35, 105), (30, -90), (-5, 120), (15, -90), (50, 30),
        (35, 139), (28, 77), (-33, 151), (37, -122), (52, 13),
        # Add more diverse regions
        (-30, -60), (60, 10), (0, 0), (40, 140), (-40, 175)
    ]
    
    samples_per_region = n_samples // len(regions)
    coordinates = []
    
    for lat, lon in regions:
        # Add more variation within each region
        region_samples = np.random.normal(
            loc=[lat, lon],
            scale=[2, 2],  # Increased variation
            size=(samples_per_region, 2)
        )
        coordinates.extend(region_samples)
    
    return np.array(coordinates)

def generate_realistic_land_use(n_samples):
    """Generate more realistic land use distribution"""
    # Updated proportions based on global land use statistics
    land_use_probs = {
        0: 0.31,  # Forest
        1: 0.38,  # Agriculture
        2: 0.24,  # Grassland
        3: 0.02,  # Bare Soil
        4: 0.05   # Urban
    }
    
    return np.random.choice(
        list(land_use_probs.keys()),
        size=n_samples,
        p=list(land_use_probs.values())
    )

def generate_enhanced_climate_data(n_samples, coordinates):
    """Generate enhanced climate data with more extreme conditions"""
    # Base climate data
    temperature = np.random.normal(20, 15, n_samples)  # Wider temperature range
    humidity = np.random.normal(60, 20, n_samples)    # More humidity variation
    
    # Add extreme conditions
    extreme_mask = np.random.random(n_samples) < 0.1  # 10% extreme conditions
    temperature[extreme_mask] = np.random.choice(
        [-40, 50],  # Extreme temperatures
        size=extreme_mask.sum()
    )
    humidity[extreme_mask] = np.random.choice(
        [10, 95],   # Extreme humidity
        size=extreme_mask.sum()
    )
    
    # Enhanced rainfall erosivity
    rainfall = np.random.gamma(2, 50, n_samples)  # More realistic distribution
    rainfall[extreme_mask] *= 3  # Extreme rainfall events
    
    return temperature, humidity, rainfall

def generate_enhanced_soil_properties(n_samples):
    """Generate enhanced soil properties with more variation"""
    # Enhanced organic matter distribution
    organic_matter = np.random.beta(2, 2, n_samples) * 6  # 0-6% range
    
    # Enhanced soil depth
    soil_depth = np.random.gamma(2, 0.5, n_samples)  # 0-5m range
    
    # Enhanced soil texture
    textures = ['Clayey', 'Sandy', 'Loamy', 'Silty', 'Peaty']
    texture_probs = [0.25, 0.25, 0.25, 0.15, 0.10]
    soil_texture = np.random.choice(textures, size=n_samples, p=texture_probs)
    
    return organic_matter, soil_depth, soil_texture

def generate_seasonal_data(n_samples):
    """Generate enhanced seasonal data with more variation"""
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    # Slightly biased towards growing seasons
    season_probs = [0.22, 0.28, 0.28, 0.22]
    return np.random.choice(seasons, size=n_samples, p=season_probs)

def calculate_erosion_rate(row):
    """Enhanced erosion rate calculation"""
    # Base factors
    R = row['rainfall_erosivity']
    K = 0.2 + 0.3 * row['soil_organic_matter'] / 5  # Soil erodibility
    L = row['slope_length']
    S = row['slope_steepness']
    
    # Enhanced land use factor
    C = {
        0: 0.001,  # Forest
        1: 0.3,    # Agriculture
        2: 0.1,    # Grassland
        3: 0.8,    # Bare Soil
        4: 0.01    # Urban
    }[row['land_use']]
    
    # Seasonal adjustment
    season_factor = {
        'Winter': 0.5,
        'Spring': 1.2,
        'Summer': 0.8,
        'Fall': 1.0
    }[row['season']]
    
    # Calculate erosion rate
    erosion = R * K * L * S * C * season_factor
    
    # Add some random variation
    erosion *= np.random.normal(1, 0.1)
    
    return max(0, erosion)

def generate_improved_dataset(n_samples=50000):
    """Generate improved dataset with balanced distributions"""
    # Generate coordinates
    coordinates = generate_balanced_geography(n_samples)
    
    # Generate features
    land_use = generate_realistic_land_use(n_samples)
    temperature, humidity, rainfall = generate_enhanced_climate_data(n_samples, coordinates)
    organic_matter, soil_depth, soil_texture = generate_enhanced_soil_properties(n_samples)
    seasons = generate_seasonal_data(n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'latitude': coordinates[:, 0],
        'longitude': coordinates[:, 1],
        'elevation_m': np.random.normal(500, 300, n_samples),
        'temperature_c': temperature,
        'humidity_percent': humidity,
        'rainfall_erosivity': rainfall,
        'soil_organic_matter': organic_matter,
        'soil_depth': soil_depth,
        'soil_texture': soil_texture,
        'land_use': land_use,
        'season': seasons,
        'slope_length': np.random.uniform(10, 100, n_samples),
        'slope_steepness': np.random.uniform(0, 45, n_samples)
    })
    
    # Calculate erosion rate
    df['erosion_rate'] = df.apply(calculate_erosion_rate, axis=1)
    
    return df

def analyze_improved_dataset(df):
    """Analyze the improved dataset"""
    print("\nImproved Dataset Analysis:")
    
    # Geographic distribution
    print("\n1. Geographic Distribution:")
    region_counts = df.groupby(['latitude', 'longitude']).size()
    print(region_counts)
    
    # Land use distribution
    print("\n2. Land Use Distribution:")
    land_use_dist = df['land_use'].value_counts(normalize=True) * 100
    print(land_use_dist)
    
    # Climate statistics
    print("\n3. Climate Statistics:")
    print(df[['temperature_c', 'humidity_percent', 'rainfall_erosivity']].describe())
    
    # Soil properties
    print("\n4. Soil Properties:")
    print("\nOrganic Matter:")
    print(df['soil_organic_matter'].describe())
    print("\nSoil Depth:")
    print(df['soil_depth'].describe())
    print("\nSoil Texture Distribution:")
    print(df['soil_texture'].value_counts())
    
    # Seasonal distribution
    print("\n5. Seasonal Distribution:")
    season_dist = df['season'].value_counts(normalize=True) * 100
    print(season_dist)

if __name__ == "__main__":
    # Generate improved dataset
    print("Generating improved dataset...")
    df = generate_improved_dataset()
    
    # Save dataset
    df.to_csv('improved_soil_erosion_dataset.csv', index=False)
    print("\nDataset saved as 'improved_soil_erosion_dataset.csv'")
    
    # Analyze dataset
    analyze_improved_dataset(df) 