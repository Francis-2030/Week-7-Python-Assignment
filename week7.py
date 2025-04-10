# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set style for better looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

## Task 1: Load and Explore the Dataset

# Load the Iris dataset
try:
    iris = load_iris()
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                          columns=iris['feature_names'] + ['target'])
    
    # Map target numbers to species names
    iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Display first few rows
    print("First 5 rows of the dataset:")
    display(iris_df.head())
    
    # Explore structure
    print("\nDataset info:")
    iris_df.info()
    
    # Check for missing values
    print("\nMissing values per column:")
    print(iris_df.isnull().sum())
    
    # Clean data (though Iris dataset is already clean)
    # This is just to demonstrate the process
    iris_df_clean = iris_df.dropna()  # Would drop rows with missing values if any existed
    
except Exception as e:
    print(f"Error loading dataset: {e}")

## Task 2: Basic Data Analysis

# Basic statistics
print("\nBasic statistics for numerical columns:")
display(iris_df_clean.describe())

# Group by species and calculate mean measurements
print("\nMean measurements by species:")
species_stats = iris_df_clean.groupby('species').mean()
display(species_stats)

# Interesting findings
print("\nInteresting findings:")
print("- Setosa has significantly smaller petal measurements than other species")
print("- Virginica has the largest sepal length on average")
print("- Versicolor is intermediate in most measurements")

## Task 3: Data Visualization

# 1. Line chart (simulating time series - though Iris isn't temporal)
# We'll use index as pseudo-time for demonstration
plt.figure(figsize=(12, 6))
iris_df_clean['sepal length (cm)'].plot(title='Sepal Length (simulated time series)')
plt.xlabel('Observation index (pseudo-time)')
plt.ylabel('Sepal Length (cm)')
plt.show()

# 2. Bar chart - average petal length per species
plt.figure()
iris_df_clean.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Petal Length by Species')
plt.ylabel('Petal Length (cm)')
plt.xticks(rotation=0)
plt.show()

# 3. Histogram - distribution of sepal width
plt.figure()
sns.histplot(data=iris_df_clean, x='sepal width (cm)', kde=True, bins=15, color='green')
plt.title('Distribution of Sepal Width')
plt.show()

# 4. Scatter plot - sepal length vs petal length with species differentiation
plt.figure()
sns.scatterplot(data=iris_df_clean, x='sepal length (cm)', y='petal length (cm)', 
                hue='species', palette='viridis', s=100)
plt.title('Sepal Length vs Petal Length by Species')
plt.legend(title='Species')
plt.show()

# Bonus: Pairplot to show all relationships
plt.figure()
sns.pairplot(iris_df_clean, hue='species', palette='viridis')
plt.suptitle('Pairwise Relationships in Iris Dataset', y=1.02)
plt.show()
