"""
pom_v2.py — Enhanced Deep Learning Benchmark for PPG Classification
Author: Stevan Jokic, 2025
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, matthews_corrcoef, balanced_accuracy_score,
    cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import wilcoxon
import logging
from itertools import combinations

# === CONFIGURATION ===
MAX_EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 10
LEARNING_RATE = 1e-3

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs("results", exist_ok=True)

MODEL_STYLES = {
    'MLP': {'linestyle': '-', 'marker': 'o', 'hatch': '/', 'color': '#66c2a5'},
    'CNN': {'linestyle': '--', 'marker': '^', 'hatch': '\\', 'color': '#fc8d62'},
    'ResNet': {'linestyle': ':', 'marker': 's', 'hatch': '|', 'color': '#8da0cb'}
}

# === DATA LOADER ===
def load_signal(df, signal_col, target_length=100):
    valid_signals, valid_indices, invalid = [], [], []
    for idx, (app_id, signal_str) in enumerate(zip(df['app_id'], df[signal_col])):
        try:
            if not isinstance(signal_str, str) or not signal_str.strip():
                continue
            signal_list = [float(x) for x in signal_str.split(',') if x.strip()]
            if len(signal_list) < target_length:
                signal_list = np.pad(signal_list, (0, target_length - len(signal_list)), mode='edge')
            elif len(signal_list) > target_length:
                signal_list = signal_list[:target_length]
            signal_array = np.array(signal_list, dtype=np.float32)
            valid_signals.append(signal_array)
            valid_indices.append(idx)
        except Exception as e:
            invalid.append((app_id, str(e)))
            continue
    if invalid:
        logging.warning(f"Excluded {len(invalid)} invalid signals for {signal_col}")
    return np.stack(valid_signals), valid_indices

# === MODEL DEFINITIONS ===
class OptimizedMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.network(x)

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.shortcut = nn.Conv1d(1, 128, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out = self.features(x)
        shortcut = self.shortcut(x)
        out = out + shortcut.mean(dim=2, keepdim=True)
        return self.classifier(out)

# --- SE-ResNet ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)

class OptimizedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride, 1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_c)
        self.proj = nn.Sequential(nn.Conv1d(in_c, out_c, 1, stride), nn.BatchNorm1d(out_c)) \
                    if in_c != out_c or stride != 1 else None
    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.proj is not None: res = self.proj(res)
        return self.relu(out + res)

class OptimizedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.init = nn.Sequential(nn.Conv1d(1, 32, 7, 1, 3), nn.BatchNorm1d(32), nn.ReLU())
        self.layer1 = OptimizedResidualBlock(32, 32)
        self.layer2 = OptimizedResidualBlock(32, 64, 2)
        self.layer3 = OptimizedResidualBlock(64, 128, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        x = self.init(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x)
        return self.classifier(x)

# === TRAINING ===
def train_model(model, X, y, device, model_name, results_dir, num_classes):
    # Fix for class weights computation
    y_np = y.cpu().numpy()
    classes = np.arange(num_classes)  # Use all possible classes
    try:
        class_weights = compute_class_weight('balanced', classes=classes, y=y_np)
    except ValueError:
        # Fallback: use uniform weights if computation fails
        logging.warning(f"Class weight computation failed for {model_name}, using uniform weights")
        class_weights = np.ones(num_classes)
    
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    
    val_size = int(0.2 * len(X))
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    best_loss = np.inf
    patience = 0
    train_losses, val_losses = [], []
    start = time.time()
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device)).item()
        
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), f"{results_dir}/classification_dl_{model_name}_best.pth")
        else:
            patience += 1
            if patience >= PATIENCE: 
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(f"{results_dir}/classification_dl_{model_name}_best.pth"))
    
    # Save learning curve data
    learning_data = pd.DataFrame({
        'epoch': range(len(train_losses)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    learning_data.to_csv(f"{results_dir}/classification_dl_{model_name}_learning_curve_data.csv", index=False)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Learning Curve - {model_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{results_dir}/classification_dl_{model_name}_learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    return model, time.time() - start

# === PREDICTION ===
def predict_model(X, model, device, is_cnn=False):
    model.eval()
    start = time.time()
    if is_cnn: 
        X = X.unsqueeze(1)
    X = X.to(device)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs, time.time() - start

# === COMPREHENSIVE EVALUATION ===
def evaluate(model_name, signal_name, fold, y_true, y_pred, y_proba, t_train, t_pred, out_dir):
    # Ensure y_true is numpy array
    y_true = np.array(y_true)
    
    # Calculate comprehensive metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred)
    }
    
    # Handle ROC AUC calculation (can fail for some edge cases)
    try:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        if y_proba.shape[1] == len(np.unique(y_true)):
            metrics['roc_auc'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        else:
            metrics['roc_auc'] = np.nan
    except Exception as e:
        logging.warning(f"ROC AUC calculation failed for {model_name}, {signal_name}, fold {fold}: {e}")
        metrics['roc_auc'] = np.nan
    
    # Save ROC curve data (if possible)
    try:
        if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
            roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
            roc_data.to_csv(f"{out_dir}/classification_dl_{signal_name}_{model_name}_fold{fold}_roc_data.csv", index=False)
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color=MODEL_STYLES[model_name]['color'], 
                     linestyle=MODEL_STYLES[model_name]['linestyle'], linewidth=2,
                     label=f'{model_name} (AUC = {metrics["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {model_name} ({signal_name}, Fold {fold})', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{out_dir}/classification_dl_{signal_name}_{model_name}_fold{fold}_roc_curve.png", dpi=300, bbox_inches="tight")
            plt.close()
    except Exception as e:
        logging.warning(f"ROC curve generation failed for {model_name}, {signal_name}, fold {fold}: {e}")
    
    # Save Precision-Recall curve data
    try:
        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_proba.ravel())
        pr_data = pd.DataFrame({'precision': precision, 'recall': recall})
        pr_data.to_csv(f"{out_dir}/classification_dl_{signal_name}_{model_name}_fold{fold}_pr_data.csv", index=False)
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color=MODEL_STYLES[model_name]['color'],
                 linestyle=MODEL_STYLES[model_name]['linestyle'], linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name} ({signal_name}, Fold {fold})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/classification_dl_{signal_name}_{model_name}_fold{fold}_pr_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logging.warning(f"PR curve generation failed for {model_name}, {signal_name}, fold {fold}: {e}")
    
    # Save calibration curve data
    try:
        prob_true, prob_pred = calibration_curve(y_true_bin.ravel(), y_proba.ravel(), n_bins=10)
        calibration_data = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
        calibration_data.to_csv(f"{out_dir}/classification_dl_{signal_name}_{model_name}_fold{fold}_calibration_data.csv", index=False)
        
        # Plot calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', color=MODEL_STYLES[model_name]['color'],
                 linestyle=MODEL_STYLES[model_name]['linestyle'], linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title(f'Calibration Plot - {model_name} ({signal_name}, Fold {fold})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/classification_dl_{signal_name}_{model_name}_fold{fold}_calibration_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logging.warning(f"Calibration plot generation failed for {model_name}, {signal_name}, fold {fold}: {e}")
    
    # Save confusion matrix data
    cm = confusion_matrix(y_true, y_pred)
    cm_data = pd.DataFrame(cm)
    cm_data.to_csv(f"{out_dir}/classification_dl_{signal_name}_{model_name}_fold{fold}_confusion_matrix.csv", index=False)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name} ({signal_name}, Fold {fold})', fontsize=14)
    plt.savefig(f"{out_dir}/classification_dl_{signal_name}_{model_name}_fold{fold}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Generate detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{out_dir}/classification_dl_{signal_name}_{model_name}_fold{fold}_classification_report.csv")
    
    return {**metrics, 'train_time': t_train, 'pred_time': t_pred, 'model': model_name, 
            'signal': signal_name, 'fold': fold}

# === STATISTICAL ANALYSIS ===
def perform_statistical_analysis(summary_df, results_dir):
    """Perform Wilcoxon signed-rank tests for model comparisons"""
    try:
        models = summary_df['model'].unique()
        signals = summary_df['signal'].unique()
        
        with open(f"{results_dir}/classification_dl_statistical_analysis.txt", "w") as f:
            f.write("Statistical Analysis Results (Wilcoxon Signed-Rank Test)\n")
            f.write("=" * 60 + "\n\n")
            
            for signal in signals:
                f.write(f"Signal: {signal}\n")
                f.write("-" * 40 + "\n")
                
                signal_data = summary_df[summary_df['signal'] == signal]
                
                for metric in ['f1_macro', 'roc_auc', 'accuracy', 'balanced_accuracy']:
                    f.write(f"\nMetric: {metric}\n")
                    
                    for model1, model2 in combinations(models, 2):
                        scores1 = signal_data[signal_data['model'] == model1][metric].values
                        scores2 = signal_data[signal_data['model'] == model2][metric].values
                        
                        # Remove NaN values for statistical test
                        mask = ~(np.isnan(scores1) | np.isnan(scores2))
                        scores1_clean = scores1[mask]
                        scores2_clean = scores2[mask]
                        
                        if len(scores1_clean) == len(scores2_clean) and len(scores1_clean) > 1:
                            try:
                                stat, p_value = wilcoxon(scores1_clean, scores2_clean)
                                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                                f.write(f"{model1} vs {model2}: W={stat:.3f}, p={p_value:.4f} {significance}\n")
                            except Exception as e:
                                f.write(f"{model1} vs {model2}: Test failed - {e}\n")
                        else:
                            f.write(f"{model1} vs {model2}: Insufficient data for comparison\n")
                
                f.write("\n")
            
            # Overall comparison across all signals
            f.write("\nOverall Comparison (All Signals Combined)\n")
            f.write("-" * 50 + "\n")
            
            for metric in ['f1_macro', 'roc_auc']:
                f.write(f"\nMetric: {metric}\n")
                
                for model1, model2 in combinations(models, 2):
                    scores1 = summary_df[summary_df['model'] == model1][metric].values
                    scores2 = summary_df[summary_df['model'] == model2][metric].values
                    
                    # Remove NaN values for statistical test
                    mask = ~(np.isnan(scores1) | np.isnan(scores2))
                    scores1_clean = scores1[mask]
                    scores2_clean = scores2[mask]
                    
                    if len(scores1_clean) == len(scores2_clean) and len(scores1_clean) > 1:
                        try:
                            stat, p_value = wilcoxon(scores1_clean, scores2_clean)
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                            f.write(f"{model1} vs {model2}: W={stat:.3f}, p={p_value:.4f} {significance}\n")
                        except Exception as e:
                            f.write(f"{model1} vs {model2}: Test failed - {e}\n")
                    else:
                        f.write(f"{model1} vs {model2}: Insufficient data for comparison\n")
    
    except Exception as e:
        logging.error(f"Error in statistical analysis: {e}")

# === COMPREHENSIVE VISUALIZATION ===
def create_comprehensive_visualizations(summary_df, results_dir):
    """Create comprehensive visualizations for publication"""
    
    # 1. Performance comparison across models and signals
    metrics_to_plot = ['f1_macro', 'roc_auc', 'accuracy', 'balanced_accuracy']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Prepare data for boxplot
        plot_data = []
        for model in summary_df['model'].unique():
            for signal in summary_df['signal'].unique():
                values = summary_df[(summary_df['model'] == model) & 
                                   (summary_df['signal'] == signal)][metric].values
                # Remove NaN values
                values = values[~np.isnan(values)]
                for val in values:
                    plot_data.append({'model': model, 'signal': signal, 'value': val})
        
        plot_df = pd.DataFrame(plot_data)
        
        if not plot_df.empty:
            sns.boxplot(data=plot_df, x='model', y='value', hue='signal', ax=ax,
                       palette=['#66c2a5', '#fc8d62'], linewidth=1.5)
            ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Signal Type', fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/classification_dl_comprehensive_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save performance comparison data
    performance_data = []
    for model in summary_df['model'].unique():
        for signal in summary_df['signal'].unique():
            for metric in metrics_to_plot:
                values = summary_df[(summary_df['model'] == model) & 
                                   (summary_df['signal'] == signal)][metric].values
                # Remove NaN values
                values = values[~np.isnan(values)]
                for val in values:
                    performance_data.append({
                        'model': model, 
                        'signal': signal, 
                        'metric': metric, 
                        'value': val
                    })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(f"{results_dir}/classification_dl_performance_comparison_data.csv", index=False)
    
    # 2. Training and prediction time analysis
    plt.figure(figsize=(12, 5))
    
    time_data = []
    for model in summary_df['model'].unique():
        for signal in summary_df['signal'].unique():
            train_times = summary_df[(summary_df['model'] == model) & 
                                   (summary_df['signal'] == signal)]['train_time'].values
            pred_times = summary_df[(summary_df['model'] == model) & 
                                  (summary_df['signal'] == signal)]['pred_time'].values
            for train_time, pred_time in zip(train_times, pred_times):
                time_data.append({'model': model, 'signal': signal, 'type': 'Training', 'time': train_time})
                time_data.append({'model': model, 'signal': signal, 'type': 'Prediction', 'time': pred_time})
    
    time_df = pd.DataFrame(time_data)
    
    if not time_df.empty:
        sns.boxplot(data=time_df, x='model', y='time', hue='type', palette=['#8da0cb', '#e78ac3'])
        plt.title('Training and Prediction Time Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/classification_dl_time_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save time analysis data
    time_df.to_csv(f"{results_dir}/classification_dl_time_analysis_data.csv", index=False)
    
    # 3. Save comprehensive summary data
    summary_stats = summary_df.groupby(['model', 'signal']).agg({
        'accuracy': ['mean', 'std'],
        'f1_macro': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'train_time': ['mean', 'std'],
        'pred_time': ['mean', 'std']
    }).round(4)
    
    summary_stats.to_csv(f"{results_dir}/classification_dl_comprehensive_summary_statistics.csv")
    summary_stats.to_latex(f"{results_dir}/classification_dl_comprehensive_summary_statistics.tex")

# === MAIN PIPELINE ===
def run_benchmark(data_path, results_dir="results", prefix="classification_dl_"):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['app_id', 'class_label'])
    df['app_id'] = pd.to_numeric(df['app_id'], errors='coerce').fillna(0).astype(int)
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    gkf = GroupKFold(5)
    all_metrics = []

    selected = [m.lower() for m in sys.argv[1:]] if len(sys.argv) > 1 else ["mlp", "cnn", "resnet"]
    logging.info(f"Running models: {selected}")

    for signal_col in ['template_ppg_norm', 'sd_template_ppg_norm']:
        if signal_col not in df.columns: 
            logging.warning(f"Signal column {signal_col} not found, skipping")
            continue
            
        target_len = 100
        X_all, valid_idx = load_signal(df, signal_col, target_len)
        y_all = pd.factorize(df.iloc[valid_idx]['class_label'])[0]
        groups = df.iloc[valid_idx]['app_id'].values
        
        num_classes = len(np.unique(y_all))
        logging.info(f"Processing {signal_col}: {len(X_all)} samples, {num_classes} classes")
        
        for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, groups)):
            logging.info(f"Fold {fold + 1}/5 for {signal_col}")
            
            X_train, y_train = torch.tensor(X_all[tr], dtype=torch.float32), torch.tensor(y_all[tr], dtype=torch.long)
            X_test, y_test = torch.tensor(X_all[te], dtype=torch.float32), y_all[te]
            input_dim = X_train.shape[1]
            
            # MLP
            if "mlp" in selected:
                logging.info("Training MLP...")
                model, t_train = train_model(OptimizedMLP(input_dim, num_classes).to(device), 
                                           X_train, y_train, device, f"mlp_{signal_col}_fold{fold}", results_dir, num_classes)
                y_pred, y_proba, t_pred = predict_model(X_test, model, device)
                all_metrics.append(evaluate("MLP", signal_col, fold, y_test, y_pred, y_proba, 
                                          t_train, t_pred, results_dir))
            
            # CNN
            if "cnn" in selected:
                logging.info("Training CNN...")
                model, t_train = train_model(ImprovedCNN(num_classes).to(device), 
                                           X_train.unsqueeze(1), y_train, device, f"cnn_{signal_col}_fold{fold}", results_dir, num_classes)
                y_pred, y_proba, t_pred = predict_model(X_test, model, device, is_cnn=True)
                all_metrics.append(evaluate("CNN", signal_col, fold, y_test, y_pred, y_proba, 
                                          t_train, t_pred, results_dir))
            
            # ResNet
            if "resnet" in selected:
                logging.info("Training ResNet...")
                model, t_train = train_model(OptimizedResNet(num_classes).to(device), 
                                           X_train.unsqueeze(1), y_train, device, f"resnet_{signal_col}_fold{fold}", results_dir, num_classes)
                y_pred, y_proba, t_pred = predict_model(X_test, model, device, is_cnn=True)
                all_metrics.append(evaluate("ResNet", signal_col, fold, y_test, y_pred, y_proba, 
                                          t_train, t_pred, results_dir))
    
    # Save all metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(f"{results_dir}/{prefix}comprehensive_results.csv", index=False)
        
        # Generate summary statistics
        summary_df = metrics_df.groupby(['model', 'signal']).agg({
            'accuracy': ['mean', 'std'],
            'f1_macro': ['mean', 'std'],
            'roc_auc': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std'],
            'precision_macro': ['mean', 'std'],
            'recall_macro': ['mean', 'std'],
            'mcc': ['mean', 'std'],
            'kappa': ['mean', 'std'],
            'train_time': ['mean', 'std'],
            'pred_time': ['mean', 'std']
        }).round(4)
        
        summary_df.to_csv(f"{results_dir}/{prefix}summary_statistics.csv")
        summary_df.to_latex(f"{results_dir}/{prefix}summary_statistics.tex")
        
        # Perform statistical analysis
        perform_statistical_analysis(metrics_df, results_dir)
        
        # Create comprehensive visualizations
        create_comprehensive_visualizations(metrics_df, results_dir)
        
        logging.info(f"Benchmark completed. Results saved to {results_dir}/")
        logging.info(f"Comprehensive results: {prefix}comprehensive_results.csv")
        logging.info(f"Summary statistics: {prefix}summary_statistics.csv")
        logging.info(f"Statistical analysis: classification_dl_statistical_analysis.txt")
    else:
        logging.error("No metrics were collected. Benchmark failed.")

if __name__ == "__main__":
    DATA_PATH = "data/train_rws_ppg_classification_dataset.csv"
    
    # Create results directory with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # RESULTS_DIR = f"results_classification_dl_benchmark_{timestamp}"
    RESULTS_DIR = f"results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save environment information
    with open(f"{RESULTS_DIR}/classification_dl_environment_info.txt", "w") as f:
        f.write(f"Deep Learning PPG Classification Benchmark\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Python: {sys.version}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA device: {torch.cuda.get_device_name()}\n")
    
    run_benchmark(DATA_PATH, RESULTS_DIR, "classification_dl_")