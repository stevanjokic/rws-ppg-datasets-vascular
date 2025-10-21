import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import os
import json

# Load dataset
df = pd.read_csv('data/rws_ppg_classification_dataset.csv')

outputFolder = "results/"
os.makedirs(outputFolder, exist_ok=True)

# File prefix for all generated files
file_prefix = "classification_dataset_"

def validate_dataset(df):
    """Comprehensive dataset validation"""
    issues = []
    
    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.any():
        issues.append(f"Missing values detected: {missing_data[missing_data > 0].to_dict()}")
    
    # Check class balance
    class_balance = df['class'].value_counts(normalize=True)
    if class_balance.min() < 0.05:  # Less than 5% in any class
        issues.append("Severe class imbalance detected")
    
    # Check physiological plausibility
    implausible_hr = df[(df['hr'] < 40) | (df['hr'] > 180)]
    if len(implausible_hr) > 0:
        issues.append(f"Implausible HR values: {len(implausible_hr)} records")
    
    return issues

# Run validation
validation_issues = validate_dataset(df)
if validation_issues:
    print("Validation warnings:")
    for issue in validation_issues:
        print(f"  - {issue}")
else:
    print("✓ Dataset validation passed")

# Filter invalid entries
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

# Statistical tests
classes_sorted = sorted(df['class'].unique())

# ANOVA for age differences between classes
age_by_class = [df[df['class'] == cls]['age'] for cls in classes_sorted]
f_val, p_val = stats.f_oneway(*age_by_class)

# Chi-square test for gender distribution across classes
gender_class_crosstab = pd.crosstab(df['gender'], df['class'])
chi2, chi_p, dof, expected = stats.chi2_contingency(gender_class_crosstab)

# Improved summary tables
demographic_summary = pd.DataFrame({
    'Variable': ['Participants', 'Records', 'Age (years)', 'Heart Rate (bpm)', 'Sex (M/F)'],
    'Total': [
        num_users,
        num_records,
        f"{age_mean:.1f} ± {age_std:.1f}",
        f"{hr_mean:.1f} ± {hr_std:.1f}", 
        f"{num_male}/{num_female}"
    ],
    'Range/Details': [
        '',
        '',
        f"{age_range[0]}-{age_range[1]}",
        f"{hr_range[0]}-{hr_range[1]}",
        f"({num_male/num_users*100:.1f}%/{num_female/num_users*100:.1f}%)"
    ]
})

class_distribution = pd.DataFrame({
    'Class': class_counts.index,
    'Percentage': [f"{val:.1f}%" for val in class_counts.values],
    'Count': [df[df['class'] == cls].shape[0] for cls in class_counts.index]
})

# Legacy summary table for compatibility
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

print(f"""
=== DATASET CHARACTERISTICS ===
Participants: {num_users:,}
Records: {num_records:,}
Age: {age_mean:.1f} ± {age_std:.1f} years ({age_range[0]}-{age_range[1]})
Heart Rate: {hr_mean:.1f} ± {hr_std:.1f} bpm ({hr_range[0]}-{hr_range[1]})
Sex: {num_male} male, {num_female} female

=== CLASS DISTRIBUTION ===
{class_distribution.to_string(index=False)}

=== STATISTICAL TESTS ===
Age differences (ANOVA): F={f_val:.3f}, p={p_val:.4f}
Gender distribution (χ²): χ²={chi2:.3f}, p={chi_p:.4f}
""")

# Save all tables
demographic_summary.to_csv(outputFolder + file_prefix + "demographic_summary.csv", index=False)
class_distribution.to_csv(outputFolder + file_prefix + "class_distribution.csv", index=False)
summary.to_csv(outputFolder + file_prefix + "statistical_summary.csv", index=False)

# Save LaTeX formatted tables
with open(outputFolder + file_prefix + "demographic_table.tex", "w") as f:
    f.write(demographic_summary.to_latex(index=False, escape=False))
    
with open(outputFolder + file_prefix + "class_distribution_table.tex", "w") as f:
    f.write(class_distribution.to_latex(index=False, escape=False))

# Create analysis metadata
metadata = {
    'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_version': '1.0',
    'total_participants': num_users,
    'total_records': num_records,
    'age_range': f"{age_range[0]}-{age_range[1]}",
    'inclusion_criteria': 'age ≥ 15, HR 45-105 bpm',
    'statistical_tests': {
        'anova_age_f': f_val,
        'anova_age_p': p_val,
        'chi2_gender_chi2': chi2,
        'chi2_gender_p': chi_p
    }
}

# Save metadata
with open(outputFolder + file_prefix + "analysis_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

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
plt.grid(True, linestyle='--', linewidth=0.5, axis='y', color='gray')
plt.legend(title="Class", fontsize=10, title_fontsize=11)
plt.tight_layout()
plt.savefig(outputFolder + file_prefix + "smooth_normalized_age_distribution_bw.png", dpi=300, format='png', bbox_inches='tight')
plt.show()

# Correlation matrix
correlation_matrix = df[['age', 'gender', 'hr', 'class']].corr(method='spearman')

plt.figure(figsize=(5, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='gray_r', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlations', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(outputFolder + file_prefix + "correlation_heatmap_bw.png", 
            dpi=300, format='png', bbox_inches='tight')
plt.show()

print("All plots and statistics have been saved to:", outputFolder)
print("File prefix:", file_prefix)
print("Analysis completed successfully!")