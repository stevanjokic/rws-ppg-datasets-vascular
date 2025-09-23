import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv('data/rws_ppg_regression_dataset.csv')

# Create output folder for images
outputFolder = "results/"
os.makedirs(outputFolder, exist_ok=True)

# Filter invalid data
df = df[df['age'] >= 15]
df = df[(df['hr'] >= 45) & (df['hr'] <= 105)]

# File prefix for all generated files
file_prefix = "regression_dataset_"

# Basic statistics for specific columns only
target_columns = ['age', 'sex', 'hr', 'rmssd']
stats_list = []

for col in target_columns:
    if col in df.columns:
        stats_list.append({
            'Variable': col,
            'Mean': df[col].mean(),
            'Std': df[col].std(),
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Median': df[col].median(),
            'Q1': df[col].quantile(0.25),
            'Q3': df[col].quantile(0.75)
        })

stats_df = pd.DataFrame(stats_list)
print("Statistical Summary for Selected Variables:")
print(stats_df.to_string(index=False))

# Save table to CSV
stats_df.to_csv(outputFolder + file_prefix + "stats_summary.csv", index=False)

# Visualizations in black-and-white scheme for printing
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['lines.color'] = 'black'

# 1. Age distribution
plt.figure(figsize=(6, 4))
plt.hist(df['age'], bins=20, color='lightgray', edgecolor='black', linewidth=0.5)
plt.title('Age Distribution', fontsize=14, weight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', linewidth=0.5, axis='y')
plt.tight_layout()
plt.savefig(outputFolder + file_prefix + "age_distribution_bw.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Heart rate distribution
plt.figure(figsize=(6, 4))
plt.hist(df['hr'], bins=20, color='lightgray', edgecolor='black', linewidth=0.5)
plt.title('Heart Rate Distribution', fontsize=14, weight='bold')
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', linewidth=0.5, axis='y')
plt.tight_layout()
plt.savefig(outputFolder + file_prefix + "hr_distribution_bw.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. RMSSD distribution
if 'rmssd' in df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df['rmssd'], bins=20, color='lightgray', edgecolor='black', linewidth=0.5)
    plt.title('RMSSD Distribution', fontsize=14, weight='bold')
    plt.xlabel('RMSSD (ms)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', linewidth=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(outputFolder + file_prefix + "rmssd_distribution_bw.png", dpi=300, bbox_inches='tight')
    plt.show()

# 4. Sex distribution
if 'sex' in df.columns:
    sex_counts = df['sex'].value_counts()
    plt.figure(figsize=(5, 4))
    bars = plt.bar(sex_counts.index.astype(str), sex_counts.values, 
                   color='lightgray', edgecolor='black', linewidth=1)
    plt.title('Sex Distribution', fontsize=14, weight='bold')
    plt.xlabel('Sex (0=Male, 1=Female)')
    plt.ylabel('Count')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height}', 
                 ha='center', va='bottom', fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(outputFolder + file_prefix + "sex_distribution_bw.png", dpi=300, bbox_inches='tight')
    plt.show()

print("All plots have been saved to:", outputFolder)
print("File prefix:", file_prefix)
print("Analysis completed successfully!")