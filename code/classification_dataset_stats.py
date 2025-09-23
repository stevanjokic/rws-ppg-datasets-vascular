import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

# Load dataset
df = pd.read_csv('data/rws_ppg_classification_dataset.csv')

outputFolder = "results/"
os.makedirs(outputFolder, exist_ok=True)

# File prefix for all generated files
file_prefix = "classification_dataset_"

# Filter invalid entries
# df = df[df['app_id'] != 0]
df = df[df['age'] >= 15]
df = df[(df['hr'] >= 45) & (df['hr'] <= 105)]

# Number of unique users
num_users = df['app_id'].nunique()

# Age statistics
age_mean = df['age'].mean()
age_std = df['age'].std()
age_range = (df['age'].min(), df['age'].max())
age_median = df['age'].median()
age_q1 = df['age'].quantile(0.25)
age_q3 = df['age'].quantile(0.75)

# Heart rate statistics
hr_mean = df['hr'].mean()
hr_std = df['hr'].std()
hr_range = (df['hr'].min(), df['hr'].max())
hr_median = df['hr'].median()
hr_q1 = df['hr'].quantile(0.25)
hr_q3 = df['hr'].quantile(0.75)

# Gender: 0 = male, 1 = female
gender_counts = df['gender'].value_counts()
num_male = gender_counts.get(0, 0)
num_female = gender_counts.get(1, 0)

# Class distribution (percentage)
class_counts = df['class'].value_counts(normalize=True).sort_index() * 100

num_records = len(df)

# Summary table
summary = pd.DataFrame({
    "Category": ["Age (years)", "", "", "Heart Rate (bpm)", "", "", "Sex (Male / Female)", "Users", "Records", "Class 1", "Class 2", "Class 3", "Class 4"],
    "Statistic": ["Mean ± SD", "Range", "Median [IQR]",
                  "Mean ± SD", "Range", "Median [IQR]",
                  "", "", "", "Distribution", "Distribution", "Distribution", "Distribution"],
    "Value": [f"{age_mean:.1f} ± {age_std:.1f}", f"{age_range[0]}–{age_range[1]}", f"{age_median:.0f} [{age_q1:.0f}-{age_q3:.0f}]",
              f"{hr_mean:.1f} ± {hr_std:.1f}", f"{hr_range[0]}–{hr_range[1]}", f"{hr_median:.0f} [{hr_q1:.0f}-{hr_q3:.0f}]",
              f"{num_male} / {num_female}", f"{num_users}", f"{num_records}",
              f"{class_counts.get(1, 0):.1f}%", f"{class_counts.get(2, 0):.1f}%", f"{class_counts.get(3, 0):.1f}%", f"{class_counts.get(4, 0):.1f}%"]
})

print("\nStatistical Summary:")
print(summary.to_string(index=False))

# Save summary table to CSV
summary.to_csv(outputFolder + file_prefix + "statistical_summary.csv", index=False)

# Black-and-white bar chart for publication (DPI 300)
plt.figure(figsize=(6, 4))
bars = plt.bar(class_counts.index.astype(str), class_counts.values, color='lightgray', edgecolor='black', linewidth=1)

# Adjust y-axis limit
max_height = max([bar.get_height() for bar in bars])
plt.ylim(0, max_height + 5)

# Annotate bars with percentages
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1.5, f"{height:.1f}%", ha='center', va='bottom', fontsize=9)

# Final chart formatting
plt.title("Class Distribution", fontsize=14, weight='bold')
plt.xlabel("Class Label", fontsize=12)
plt.ylabel("Percentage", fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, axis='y', color='gray')
plt.tight_layout()
plt.savefig(outputFolder + file_prefix + "class_distribution_bw.png", dpi=300, format='png', bbox_inches='tight')
plt.show()

# Age histograms by class
classes_sorted = sorted(df['class'].unique())
fig, axes = plt.subplots(1, len(classes_sorted), figsize=(14, 4), sharey=True)

for i, cls in enumerate(classes_sorted):
    age_data = df[df['class'] == cls]['age']
    axes[i].hist(age_data, bins=10, color='gray', edgecolor='black')
    axes[i].set_title(f"Class {cls}", fontsize=12, weight='bold')
    axes[i].set_xlabel("Age", fontsize=10)
    if i == 0:
        axes[i].set_ylabel("Count", fontsize=10)
    axes[i].grid(True, linestyle='--', linewidth=0.5, axis='y', color='gray')

plt.suptitle("Age Distribution by Class", fontsize=14, weight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(outputFolder + file_prefix + "age_distribution_by_class_bw.png", dpi=300, format='png', bbox_inches='tight')
plt.show()

# Normalized age distribution by class (5-year bins)
age_bins = np.arange(15, 85, 5)
bin_centers = age_bins[:-1] + 2.5
line_styles = ['-', '--', '-.', ':']

plt.figure(figsize=(7, 5))

for idx, cls in enumerate(classes_sorted):
    age_values = df[df['class'] == cls]['age']
    counts, _ = np.histogram(age_values, bins=age_bins)
    total = counts.sum()
    if total > 0:
        normalized = (counts / total) * 100
        plt.plot(bin_centers, normalized, linestyle=line_styles[idx % len(line_styles)],
                 color='black', marker='o', label=f"Class {cls}")

plt.xlabel("Age (years)", fontsize=12)
plt.ylabel("Frequency (%)", fontsize=12)
plt.title("Normalized Age Distribution by Class", fontsize=14, weight='bold')
plt.grid(True, linestyle='--', linewidth=0.5, axis='y', color='gray')
plt.legend(title="Class", fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig(outputFolder + file_prefix + "normalized_age_distribution_by_class_bw.png", dpi=300, format='png', bbox_inches='tight')
plt.show()

# Smoothed and normalized age distribution (5-year bins)
age_min, age_max = df['age'].min(), df['age'].max()
age_bins = np.arange(age_min, age_max + 5, 5)
bin_centers = age_bins[:-1] + 1.5

plt.figure(figsize=(7, 5))

for idx, cls in enumerate(classes_sorted):
    age_vals = df[df['class'] == cls]['age']
    counts, _ = np.histogram(age_vals, bins=age_bins)
    
    # Apply Gaussian smoothing
    smooth_counts = gaussian_filter1d(counts, sigma=1.35)
    
    # Normalize to [0, 1]
    normalized = smooth_counts / np.max(smooth_counts)
    
    # Plot without markers
    plt.plot(bin_centers, normalized,
             linestyle=line_styles[idx % len(line_styles)],
             color='black', label=f"Class {cls}")

plt.xlabel("Age (years)", fontsize=12)
plt.ylabel("Normalized Frequency", fontsize=12)
# plt.title("Smoothed and Normalized Age Distribution by Class", fontsize=14, weight='bold')
plt.grid(True, linestyle='--', linewidth=0.5, axis='y', color='gray')
plt.legend(title="Class", fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig(outputFolder + file_prefix + "smooth_normalized_age_distribution_bw.png", dpi=300, format='png', bbox_inches='tight')
plt.show()

print("All plots and statistics have been saved to:", outputFolder)
print("File prefix:", file_prefix)
print("Analysis completed successfully!")