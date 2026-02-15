# This script implements a machine learning pipeline for evaluating classifiers on the RWS PPG classification dataset. It performs the following tasks:
# - Loads and preprocesses the dataset, using existing features.
# - Conducts exploratory data analysis (EDA), generating summary statistics, class distribution, and feature correlation plots.
# - Evaluates RandomForest, XGBoost, ExtraTrees, and MLP classifiers with hyperparameter tuning using GridSearchCV.
# - Applies class balancing methods (none, SMOTE, RandomUnderSampler, SMOTEENN, SMOTE-Tomek) to handle imbalanced data.
# - Uses GroupKFold cross-validation (5 folds) to prevent data leakage across groups (app_id).
# - Computes comprehensive metrics (accuracy, precision, recall, F1-score, MCC, balanced accuracy, ROC-AUC) with bootstrap confidence intervals.
# - Performs permutation importance analysis to assess feature contributions.
# - Tracks training time (including hyperparameter tuning) and prediction time per model, fold, and method.
# - Generates visualizations: ROC curves, precision-recall curves, calibration plots, confusion matrices, per-class F1-scores, summary boxplots (including timing boxplots), and feature importance, using a colorblind-friendly palette (Set2) with distinct line styles, markers, and hatches for monochrome print compatibility.
# - Conducts statistical comparisons using Wilcoxon tests to evaluate model performance differences.
# - Aggregates metrics across folds, reporting mean and standard deviation (including for training and prediction times).
# - Saves results as CSV, LaTeX tables, and high-resolution PNG plots for reproducibility and publication.
# 
# Before run this script unpack archive data/rws_ppg_classification_dataset.zip to data/rws_ppg_classification_dataset.csv and run classification_prepare_trainig_data.py 
# to generate training data data/train_rws_ppg_classification_dataset.csv
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from scipy.stats import wilcoxon
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

# Model definitions with hyperparameter tuning
MODEL_DEFS = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(eval_metric='mlogloss', random_state=42),
        'param_grid': {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15]
        }
    },
    'MLP': {
        'model': MLPClassifier(random_state=42, max_iter=1000),
        'param_grid': {
            'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
            'learning_rate_init': [0.001, 0.01],
            'alpha': [0.0001, 0.001]
        }
    }
}

# Define line styles, markers, hatches, and colors for differentiation
MODEL_STYLES = {
    'RandomForest': {'linestyle': '-', 'marker': 'o', 'hatch': '/', 'color': '#66c2a5'},  # Green
    'XGBoost': {'linestyle': '--', 'marker': '^', 'hatch': '\\', 'color': '#fc8d62'},     # Orange
    'ExtraTrees': {'linestyle': ':', 'marker': 's', 'hatch': '|', 'color': '#8da0cb'},    # Blue
    'MLP': {'linestyle': '-.', 'marker': '*', 'hatch': '-', 'color': '#e78ac3'}           # Purple
}

# Balancing methods
BALANCE_METHODS = {
    'none': lambda X, y: (X, y),
    'smote': lambda X, y: SMOTE(random_state=42).fit_resample(X, y),
    'undersample': lambda X, y: RandomUnderSampler(random_state=42).fit_resample(X, y),
    'smoteenn': lambda X, y: SMOTEENN(random_state=42).fit_resample(X, y),
    'smote_tomek': lambda X, y: SMOTETomek(random_state=42).fit_resample(X, y)
}

# Load data with error handling and data validation
def load_data(path):
    try:
        df = pd.read_csv(path)
        label_col = 'class_label'
        group_col = 'app_id'
        feature_cols = ['hr', 'amp1_2g', 'mean1_2g', 'sigma1_2g', 'amp2_2g', 'mean2_2g', 'sigma2_2g', 'SPDP', 'AIx']
        
        if not all(col in df.columns for col in [label_col, group_col] + feature_cols):
            raise ValueError("Required columns missing in dataset")
        
        # Create a copy to avoid SettingWithCopyWarning
        df_clean = df[[label_col, group_col] + feature_cols].copy()
        df_clean = df_clean.dropna(subset=[label_col, group_col] + feature_cols)
        
        # Validate and convert feature columns to numeric
        for col in feature_cols:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='raise')
            except ValueError as e:
                print(f"Error: Column {col} contains non-numeric values: {e}")
                sys.exit(1)
        
        # Normalize features
        scaler = StandardScaler()
        df_clean.loc[:, feature_cols] = scaler.fit_transform(df_clean[feature_cols])
        
        return df_clean[feature_cols], df_clean[label_col], df_clean[group_col]
    except FileNotFoundError:
        print(f"Error: File {path} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# Exploratory Data Analysis
def exploratory_data_analysis(df, results_dir):
    try:
        os.makedirs(results_dir, exist_ok=True)
        # Summary statistics
        summary_stats = df.describe()
        summary_stats.to_csv(f"{results_dir}/dataset_summary_stats.csv")
        
        # Class distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='class_label', hue='class_label', data=df, palette='Set2', edgecolor='black', hatch='/', legend=False)
        plt.title('Class Distribution', fontsize=14)
        plt.xlabel('Class Label', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.savefig(f"{results_dir}/class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature correlations (only numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='viridis', fmt='.2f', annot_kws={'size': 10})
        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.savefig(f"{results_dir}/feature_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error in EDA: {e}")

# Bootstrap confidence intervals
def bootstrap_ci(y_true, y_pred, y_proba, n_boot=1000, alpha=0.95):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    f1_scores, auc_scores = [], []
    for _ in range(n_boot):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        f1 = f1_score(y_true[idx], y_pred[idx], average='macro')
        auc = roc_auc_score(y_true_bin[idx], y_proba[idx], average='macro', multi_class='ovr')
        f1_scores.append(f1)
        auc_scores.append(auc)

    f1_ci = np.percentile(f1_scores, [(1-alpha)/2*100, (1+alpha)/2*100])
    auc_ci = np.percentile(auc_scores, [(1-alpha)/2*100, (1+alpha)/2*100])
    return f1_ci, auc_ci

# Feature importance analysis
def feature_importance_analysis(model, X, y, feature_cols, results_dir, model_name, fold, method, prefix):
    try:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': result.importances_mean,
            'std': result.importances_std
        })
        importance_df.to_csv(f"{results_dir}/{prefix}{method}_{model_name}_feature_importance_fold{fold}.csv")
        plt.figure(figsize=(8, 6))
        sns.barplot(x='importance', y='feature', data=importance_df, color=MODEL_STYLES[model_name]['color'],
                    edgecolor='black', hatch=MODEL_STYLES[model_name]['hatch'])
        plt.title(f'Feature Importance ({model_name}, {method}, Fold {fold})', fontsize=14)
        plt.xlabel('Permutation Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.savefig(f"{results_dir}/{prefix}{method}_{model_name}_feature_importance_fold{fold}.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error in feature importance: {e}")

# Evaluation and export
def evaluate_and_export(models, X_test, y_test, fold, method, results_dir, prefix):
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    summary = []
    classwise = []

    # ROC and Precision-Recall plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name, model in models.items():
        try:
            # Measure prediction time
            start_pred = time.time()
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            prediction_time = time.time() - start_pred

            f1_ci, auc_ci = bootstrap_ci(y_test, y_pred, y_proba)

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
                'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'mcc': matthews_corrcoef(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'roc_auc_ovr': roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr'),
                'report': classification_report(y_test, y_pred, output_dict=True),
                'confusion': confusion_matrix(y_test, y_pred)
            }

            summary.append({
                'fold': fold,
                'model': name,
                'method': method,
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_ci_lower': f1_ci[0],
                'f1_ci_upper': f1_ci[1],
                'precision_weighted': metrics['precision_weighted'],
                'recall_weighted': metrics['recall_weighted'],
                'f1_weighted': metrics['f1_weighted'],
                'mcc': metrics['mcc'],
                'balanced_accuracy': metrics['balanced_accuracy'],
                'roc_auc_ovr': metrics['roc_auc_ovr'],
                'auc_ci_lower': auc_ci[0],
                'auc_ci_upper': auc_ci[1],
                'training_time': models[name].__dict__.get('training_time', np.nan),
                'prediction_time': prediction_time
            })

            report_df = pd.DataFrame(metrics['report']).transpose()
            for cls in report_df.index:
                if cls.isdigit():
                    classwise.append({
                        'fold': fold,
                        'model': name,
                        'method': method,
                        'class': int(cls),
                        'precision': report_df.loc[cls, 'precision'],
                        'recall': report_df.loc[cls, 'recall'],
                        'f1': report_df.loc[cls, 'f1-score']
                    })

            # Export classification report
            report_df.to_csv(f"{results_dir}/{prefix}{method}_{name}_report_fold{fold}.csv")
            with open(f"{results_dir}/{prefix}{method}_{name}_report_fold{fold}.tex", 'w') as f:
                f.write(report_df.to_latex(float_format="%.3f"))

            # Feature importance
            feature_importance_analysis(model, X_test, y_test, X_test.columns, results_dir, name, fold, method, prefix)

            # Confusion matrix heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(metrics['confusion'], annot=True, fmt='d', cmap='viridis', annot_kws={'size': 12})
            plt.title(f'Confusion Matrix ({name}, {method}, Fold {fold})', fontsize=14)
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('True', fontsize=12)
            plt.savefig(f"{results_dir}/{prefix}{method}_{name}_confusion_fold{fold}.png", dpi=300, bbox_inches='tight')
            plt.close()

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, linestyle=MODEL_STYLES[name]['linestyle'], marker=MODEL_STYLES[name]['marker'],
                     color=MODEL_STYLES[name]['color'], label=f'{name} (AUC = {metrics["roc_auc_ovr"]:.2f})',
                     linewidth=2, markersize=8)

            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, linestyle=MODEL_STYLES[name]['linestyle'], marker=MODEL_STYLES[name]['marker'],
                     color=MODEL_STYLES[name]['color'], label=f'{name}', linewidth=2, markersize=8)

            # Calibration plot
            plt.figure(figsize=(8, 6))
            prob_true, prob_pred = calibration_curve(y_test_bin.ravel(), y_proba.ravel(), n_bins=10)
            plt.plot(prob_pred, prob_true, linestyle=MODEL_STYLES[name]['linestyle'], marker=MODEL_STYLES[name]['marker'],
                     color=MODEL_STYLES[name]['color'], label=name, linewidth=2, markersize=8)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
            plt.title(f'Calibration Plot ({name}, {method}, Fold {fold})', fontsize=14)
            plt.xlabel('Mean Predicted Probability', fontsize=12)
            plt.ylabel('Fraction of Positives', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid()
            plt.savefig(f"{results_dir}/{prefix}{method}_{name}_calibration_fold{fold}.png", dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Prediction time for {name} (Fold {fold}, {method}): {prediction_time:.2f}s")

        except Exception as e:
            print(f"Error evaluating model {name}: {e}")
            # Add NaN entry for failed models
            summary.append({
                'fold': fold,
                'model': name,
                'method': method,
                'accuracy': np.nan,
                'f1_macro': np.nan,
                'f1_ci_lower': np.nan,
                'f1_ci_upper': np.nan,
                'precision_weighted': np.nan,
                'recall_weighted': np.nan,
                'f1_weighted': np.nan,
                'mcc': np.nan,
                'balanced_accuracy': np.nan,
                'roc_auc_ovr': np.nan,
                'auc_ci_lower': np.nan,
                'auc_ci_upper': np.nan,
                'training_time': np.nan,
                'prediction_time': np.nan
            })

    # Finalize ROC and Precision-Recall plots
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.title(f'ROC Curve (Fold {fold}, {method})', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title(f'Precision-Recall Curve (Fold {fold}, {method})', fontsize=14)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{prefix}roc_pr_curve_{method}_fold{fold}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return summary, classwise

# Per-class performance visualization
def plot_classwise_performance(classwise_df, results_dir, prefix):
    try:
        if classwise_df.empty:
            print("Warning: classwise_df is empty, skipping classwise visualization")
            return
        for method in classwise_df['method'].unique():
            for model in classwise_df['model'].unique():
                subset = classwise_df[(classwise_df['method'] == method) & (classwise_df['model'] == model)]
                plt.figure(figsize=(10, 6))
                sns.barplot(x='class', y='f1', hue='fold', data=subset, palette='Set2',
                            edgecolor='black', hatch=MODEL_STYLES[model]['hatch'])
                plt.title(f'Per-Class F1-Scores ({model}, {method})', fontsize=14)
                plt.xlabel('Class', fontsize=12)
                plt.ylabel('F1-Score', fontsize=12)
                plt.legend(title='Fold', fontsize=10)
                plt.savefig(f"{results_dir}/{prefix}{method}_{model}_classwise_f1.png", dpi=300, bbox_inches='tight')
                plt.close()
    except Exception as e:
        print(f"Error in classwise visualization: {e}")

# Box plots for summary metrics (extended for timing)
def plot_summary_boxplots(summary_df, results_dir, prefix):
    try:
        if summary_df.empty:
            print("Warning: summary_df is empty, skipping summary boxplots")
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1-Macro boxplot
        for model in summary_df['model'].unique():
            subset = summary_df[summary_df['model'] == model]
            sns.boxplot(ax=axes[0, 0], x='method', y='f1_macro', hue='model', data=subset,
                        palette={m: MODEL_STYLES[m]['color'] for m in MODEL_STYLES},
                        linewidth=2, fliersize=8,
                        boxprops={'hatch': MODEL_STYLES[model]['hatch'], 'edgecolor': 'black'})
        axes[0, 0].set_title('F1-Macro Across Folds', fontsize=14)
        axes[0, 0].set_xlabel('Balancing Method', fontsize=12)
        axes[0, 0].set_ylabel('F1-Macro', fontsize=12)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend(title='Model', fontsize=10)

        # ROC-AUC boxplot
        for model in summary_df['model'].unique():
            subset = summary_df[summary_df['model'] == model]
            sns.boxplot(ax=axes[0, 1], x='method', y='roc_auc_ovr', hue='model', data=subset,
                        palette={m: MODEL_STYLES[m]['color'] for m in MODEL_STYLES},
                        linewidth=2, fliersize=8,
                        boxprops={'hatch': MODEL_STYLES[model]['hatch'], 'edgecolor': 'black'})
        axes[0, 1].set_title('ROC-AUC Across Folds', fontsize=14)
        axes[0, 1].set_xlabel('Balancing Method', fontsize=12)
        axes[0, 1].set_ylabel('ROC-AUC', fontsize=12)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(title='Model', fontsize=10)

        # Training time boxplot
        for model in summary_df['model'].unique():
            subset = summary_df[summary_df['model'] == model]
            sns.boxplot(ax=axes[1, 0], x='method', y='training_time', hue='model', data=subset,
                        palette={m: MODEL_STYLES[m]['color'] for m in MODEL_STYLES},
                        linewidth=2, fliersize=8,
                        boxprops={'hatch': MODEL_STYLES[model]['hatch'], 'edgecolor': 'black'})
        axes[1, 0].set_title('Training Time Across Folds', fontsize=14)
        axes[1, 0].set_xlabel('Balancing Method', fontsize=12)
        axes[1, 0].set_ylabel('Training Time (seconds)', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title='Model', fontsize=10)

        # Prediction time boxplot
        for model in summary_df['model'].unique():
            subset = summary_df[summary_df['model'] == model]
            sns.boxplot(ax=axes[1, 1], x='method', y='prediction_time', hue='model', data=subset,
                        palette={m: MODEL_STYLES[m]['color'] for m in MODEL_STYLES},
                        linewidth=2, fliersize=8,
                        boxprops={'hatch': MODEL_STYLES[model]['hatch'], 'edgecolor': 'black'})
        axes[1, 1].set_title('Prediction Time Across Folds', fontsize=14)
        axes[1, 1].set_xlabel('Balancing Method', fontsize=12)
        axes[1, 1].set_ylabel('Prediction Time (seconds)', fontsize=12)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(title='Model', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{results_dir}/{prefix}summary_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error in summary boxplots: {e}")

# Aggregate metrics across folds (extended for timing)
def aggregate_metrics(summary_df, results_dir, prefix):
    try:
        if summary_df.empty:
            print("Warning: summary_df is empty, skipping metric aggregation")
            return
        metrics = ['accuracy', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'mcc', 'balanced_accuracy', 'roc_auc_ovr']
        timing_metrics = ['training_time', 'prediction_time']
        agg_results = []
        for method in summary_df['method'].unique():
            for model in summary_df['model'].unique():
                subset = summary_df[(summary_df['method'] == method) & (summary_df['model'] == model)]
                agg_metrics = {
                    'method': method,
                    'model': model
                }
                for metric in metrics:
                    mean_val = subset[metric].mean()
                    std_val = subset[metric].std()
                    agg_metrics[f'{metric}_mean'] = mean_val
                    agg_metrics[f'{metric}_std'] = std_val
                for metric in timing_metrics:
                    mean_val = subset[metric].mean()
                    std_val = subset[metric].std()
                    agg_metrics[f'{metric}_mean'] = mean_val
                    agg_metrics[f'{metric}_std'] = std_val
                agg_results.append(agg_metrics)
        
        agg_df = pd.DataFrame(agg_results)
        agg_df.to_csv(f"{results_dir}/{prefix}aggregate_metrics.csv", index=False)
        
        # Generate LaTeX table for aggregated metrics
        with open(f"{results_dir}/{prefix}aggregate_metrics.tex", 'w') as f:
            f.write(agg_df.to_latex(float_format="%.3f"))
    except Exception as e:
        print(f"Error in aggregating metrics: {e}")

# Main pipeline
def run_benchmark(data_path, results_dir="results", prefix="classification_"):
    try:
        os.makedirs(results_dir, exist_ok=True)
        X, y, groups = load_data(data_path)
        
        # EDA
        df = pd.read_csv(data_path)
        exploratory_data_analysis(df, results_dir)
        
        gkf = GroupKFold(n_splits=5)
        all_records, all_classwise = [], []
        
        for method_name, balance_fn in BALANCE_METHODS.items():
            print(f"\n⚖️ Balancing method: {method_name}")
            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
                print(f"🔁 Fold {fold+1}")
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx].values
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx].values
                
                try:
                    X_bal, y_bal = balance_fn(X_train, y_train)
                    models = {}
                    for name, model_info in MODEL_DEFS.items():
                        # Measure training time
                        start_train = time.time()
                        grid = GridSearchCV(
                            model_info['model'],
                            model_info['param_grid'],
                            cv=2,
                            scoring='f1_macro',
                            n_jobs=1,
                            error_score=np.nan
                        )
                        grid.fit(X_bal, y_bal)
                        training_time = time.time() - start_train
                        
                        # Store training time in the best estimator
                        grid.best_estimator_.training_time = training_time
                        models[name] = grid.best_estimator_
                        
                        print(f"Training time for {name} (Fold {fold}, {method_name}): {training_time:.2f}s")
                    
                    fold_summary, fold_classwise = evaluate_and_export(models, X_test, y_test, fold, method_name, results_dir, prefix)
                    all_records.extend(fold_summary)
                    all_classwise.extend(fold_classwise)
                except Exception as e:
                    print(f"Error in fold {fold+1}, method {method_name}: {e}")
        
        summary_df = pd.DataFrame(all_records)
        classwise_df = pd.DataFrame(all_classwise)
        
        summary_df.to_csv(f"{results_dir}/{prefix}summary_all.csv", index=False)
        classwise_df.to_csv(f"{results_dir}/{prefix}summary_classwise.csv", index=False)
        
        # Aggregate metrics
        aggregate_metrics(summary_df, results_dir, prefix)
        
        # Additional visualizations
        plot_classwise_performance(classwise_df, results_dir, prefix)
        plot_summary_boxplots(summary_df, results_dir, prefix)
        
        with open(f"{results_dir}/environment.txt", 'w') as f:
            f.write(f"Python: {sys.version}\n")
            f.write(f"scikit-learn: {sys.modules['sklearn'].__version__}\n")
            f.write(f"xgboost: {sys.modules['xgboost'].__version__}\n")
            f.write(f"imblearn: {sys.modules['imblearn'].__version__}\n")
    
    except Exception as e:
        print(f"Error in benchmark: {e}")

# Statistical comparison using Wilcoxon test only
def statistical_comparison(summary_csv, output_txt):
    try:
        df = pd.read_csv(summary_csv)
        if df.empty:
            print(f"Warning: {summary_csv} is empty, skipping statistical comparison")
            return
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("Statistical comparison of F1-macro scores (Wilcoxon test)\n\n")
            for method in df['method'].unique():
                subset = df[df['method'] == method]
                models = subset['model'].unique()
                for i, a in enumerate(models):
                    for b in models[i+1:]:
                        f1_a = subset[subset['model'] == a]['f1_macro'].values
                        f1_b = subset[subset['model'] == b]['f1_macro'].values
                        
                        if len(f1_a) == len(f1_b) and len(f1_a) > 1:
                            w_stat, w_pval = wilcoxon(f1_a, f1_b)
                            
                            f.write(f"{method}: {a} vs {b}\n")
                            f.write(f"Wilcoxon test: W = {w_stat:.3f}, p = {w_pval:.5f}\n")
                            f.write("Interpretation:\n")
                            f.write("→ Wilcoxon: " + ("Significant difference.\n" if w_pval < 0.05 else "No significant difference.\n"))
                            f.write("\n")
    except Exception as e:
        print(f"Error in statistical comparison: {e}")

if __name__ == "__main__":
    DATA_PATH = "data/train_rws_ppg_classification_dataset.csv"
    RESULTS_DIR = "results"
    PREFIX = "classification_ensemble_"
    
    run_benchmark(DATA_PATH, RESULTS_DIR, PREFIX)
    statistical_comparison(f"{RESULTS_DIR}/{PREFIX}summary_all.csv",
                           f"{RESULTS_DIR}/{PREFIX}statistical_significance.txt")