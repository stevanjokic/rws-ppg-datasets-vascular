"""
Compares the classification performance of XGBoost models trained on two sets of features derived from Gaussian fitting of photoplethysmographic (PPG) signals. 
The first model uses parameters from two Gaussian components, while the second uses parameters from three. 
Both models are evaluated using 5-fold GroupKFold cross-validation with SMOTE-based class balancing. 
Macro-averaged F1 scores are computed for each fold, 
and statistical significance of the performance difference is assessed using paired t-tests and Wilcoxon signed-rank tests.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_rel, wilcoxon

# Load dataset
df = pd.read_csv("data/train_rws_ppg_classification_dataset.csv")  # Convert DOCX to CSV if not already done

# Define feature sets
features_2g = ['hr', 'amp1_2g', 'mean1_2g', 'sigma1_2g',
               'amp2_2g', 'mean2_2g', 'sigma2_2g']

features_3g = ['hr', 'amp1_3g', 'mean1_3g', 'sigma1_3g',
               'amp2_3g', 'mean2_3g', 'sigma2_3g',
               'amp3_3g', 'mean3_3g', 'sigma3_3g']

label_col = 'class_label'
group_col = 'app_id'

# Remove rows with missing values
df_clean = df.dropna(subset=features_2g + features_3g + [label_col, group_col])
y = df_clean[label_col]
groups = df_clean[group_col]

# Initialize evaluation
f1_scores_2g = []
f1_scores_3g = []

gkf = GroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(gkf.split(df_clean, y, groups)):
    print(f"\nFold {fold+1}")

    # Train and evaluate model using 2G features
    X_train_2g = df_clean.iloc[train_idx][features_2g]
    y_train = y.iloc[train_idx]
    X_test_2g = df_clean.iloc[test_idx][features_2g]
    y_test = y.iloc[test_idx]

    sm = SMOTE(random_state=42)
    X_train_2g_bal, y_train_bal = sm.fit_resample(X_train_2g, y_train)

    model_2g = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model_2g.fit(X_train_2g_bal, y_train_bal)
    y_pred_2g = model_2g.predict(X_test_2g)
    f1_2g = f1_score(y_test, y_pred_2g, average='macro')
    f1_scores_2g.append(f1_2g)

    # Train and evaluate model using 3G features
    X_train_3g = df_clean.iloc[train_idx][features_3g]
    X_test_3g = df_clean.iloc[test_idx][features_3g]

    X_train_3g_bal, y_train_bal = sm.fit_resample(X_train_3g, y_train)

    model_3g = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model_3g.fit(X_train_3g_bal, y_train_bal)
    y_pred_3g = model_3g.predict(X_test_3g)
    f1_3g = f1_score(y_test, y_pred_3g, average='macro')
    f1_scores_3g.append(f1_3g)

    print(f"F1 score (2G features): {f1_2g:.3f} | F1 score (3G features): {f1_3g:.3f}")

# Statistical comparison
f1_2g = np.array(f1_scores_2g)
f1_3g = np.array(f1_scores_3g)

t_stat, t_pval = ttest_rel(f1_3g, f1_2g)
w_stat, w_pval = wilcoxon(f1_3g, f1_2g)

print("\nStatistical analysis:")
print(f"Paired t-test: t = {t_stat:.3f}, p = {t_pval:.5f}")
print(f"Wilcoxon signed-rank test: W = {w_stat:.3f}, p = {w_pval:.5f}")

if t_pval < 0.05 or w_pval < 0.05:
    print("The inclusion of the third Gaussian component yields a statistically significant improvement in classification performance.")
else:
    print("There is no statistically significant difference between the 2G and 3G feature configurations.")