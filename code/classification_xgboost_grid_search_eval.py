import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Setup output directory
RESULTS_DIR = "results"
PREFIX = "classification_xgboost"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv("data/train_rws_ppg_classification_dataset.csv")

# Define feature set
features = ['hr', 'amp1_2g', 'mean1_2g', 'sigma1_2g',
            'amp2_2g', 'mean2_2g', 'sigma2_2g']
label_col = 'class_label'
group_col = 'app_id'

# Clean data
df_clean = df.dropna(subset=features + [label_col, group_col])
X = df_clean[features]
y = LabelEncoder().fit_transform(df_clean[label_col])
groups = df_clean[group_col]

# Cross-validation strategy
gkf = GroupKFold(n_splits=5)

# Pipeline
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(eval_metric='mlogloss', random_state=42))
])

# Hyperparameter grid
param_grid = {
    'xgb__n_estimators': [100, 200, 500],
    'xgb__max_depth': [3, 5, 7, 9],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__subsample': [0.8, 1.0]
}

# Grid search
grid = GridSearchCV(pipeline, param_grid=param_grid, cv=gkf, scoring='f1_macro', verbose=1, n_jobs=-1)
grid.fit(X, y, groups=groups)

# Save best parameters
best_params = grid.best_params_
pd.Series(best_params).to_csv(f"{RESULTS_DIR}/{PREFIX}_best_params.csv")
print("Best parameters:", best_params)

# Evaluation
f1_macro_scores = []
class_reports = []
per_class_f1 = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\nFold {fold+1}")

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=best_params['xgb__n_estimators'],
        max_depth=best_params['xgb__max_depth'],
        learning_rate=best_params['xgb__learning_rate'],
        subsample=best_params['xgb__subsample'],
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test)

    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_macro_scores.append(f1_macro)

    report = classification_report(y_test, y_pred, output_dict=True)
    class_reports.append(report)

    # Extract per-class F1 scores
    f1_scores = {label: metrics['f1-score'] for label, metrics in report.items() if label.isdigit()}
    per_class_f1.append(f1_scores)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - Fold {fold+1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{PREFIX}_confusion_fold{fold+1}.png")
    plt.close()

# Plot macro F1 scores
plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), f1_macro_scores, marker='o')
plt.title("Macro F1 Score Across Folds")
plt.xlabel("Fold")
plt.ylabel("F1 Macro")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/{PREFIX}_f1_macro_across_folds.png")
plt.close()

# Save averaged classification report
flat_reports = pd.DataFrame([pd.json_normalize(r).iloc[0] for r in class_reports])
avg_report = flat_reports.mean(numeric_only=True)
avg_report.to_csv(f"{RESULTS_DIR}/{PREFIX}_classification_report_avg.csv")

# Save per-class F1 scores across folds
per_class_df = pd.DataFrame(per_class_f1)
per_class_df.to_csv(f"{RESULTS_DIR}/{PREFIX}_per_class_f1_scores.csv", index=False)

# 📊 Visualize per-class variability
plt.figure(figsize=(8, 6))
sns.boxplot(data=per_class_df)
plt.title("Per-Class F1 Score Variability Across Folds")
plt.ylabel("F1 Score")
plt.xlabel("Class Label")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/{PREFIX}_per_class_f1_boxplot.png")
plt.close()

# 📈 Bootstrap confidence intervals
n_boot = 1000
boot_means = {}
boot_cis = {}

for col in per_class_df.columns:
    scores = per_class_df[col].dropna()
    boot_samples = [np.mean(np.random.choice(scores, size=len(scores), replace=True)) for _ in range(n_boot)]
    boot_means[col] = np.mean(boot_samples)
    boot_cis[col] = np.percentile(boot_samples, [2.5, 97.5])

# Save bootstrap results
boot_df = pd.DataFrame({
    'Class': list(boot_means.keys()),
    'Mean F1': list(boot_means.values()),
    'CI Lower': [ci[0] for ci in boot_cis.values()],
    'CI Upper': [ci[1] for ci in boot_cis.values()]
})
boot_df.to_csv(f"{RESULTS_DIR}/{PREFIX}_bootstrap_f1_ci.csv", index=False)

# Plot bootstrap CIs
plt.figure(figsize=(8, 6))
for i, row in boot_df.iterrows():
    plt.errorbar(x=row['Class'], y=row['Mean F1'], 
                 yerr=[[row['Mean F1'] - row['CI Lower']], [row['CI Upper'] - row['Mean F1']]],
                 fmt='o', capsize=5, label=f"Class {row['Class']}")
plt.title("Bootstrap 95% CI for Per-Class F1 Scores")
plt.xlabel("Class Label")
plt.ylabel("F1 Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/{PREFIX}_bootstrap_f1_ci_plot.png")
plt.close()

# Summary
print("\nAverage macro F1 score:", np.mean(f1_macro_scores))