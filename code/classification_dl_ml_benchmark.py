import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import label_binarize
from scipy.stats import ttest_rel, wilcoxon


# 📦 Signal loader
def load_signal(df, signal_col):
    return np.stack(df[signal_col].apply(lambda s: np.array(list(map(float, s.split(','))))).values)

# 🧠 MLP
def train_mlp(X, y, input_dim, num_classes, epochs=20, lr=1e-3):
    W1 = torch.randn(input_dim, 128, requires_grad=True)
    b1 = torch.zeros(128, requires_grad=True)
    W2 = torch.randn(128, 64, requires_grad=True)
    b2 = torch.zeros(64, requires_grad=True)
    W3 = torch.randn(64, num_classes, requires_grad=True)
    b3 = torch.zeros(num_classes, requires_grad=True)

    for _ in range(epochs):
        a1 = torch.relu(X @ W1 + b1)
        a2 = torch.relu(a1 @ W2 + b2)
        logits = a2 @ W3 + b3
        log_probs = torch.log_softmax(logits, dim=1)
        loss = torch.nn.functional.nll_loss(log_probs, y)
        loss.backward()
        with torch.no_grad():
            for param in [W1, b1, W2, b2, W3, b3]:
                param -= lr * param.grad
                param.grad.zero_()
    return [W1, b1, W2, b2, W3, b3]

def predict_mlp(X, weights):
    W1, b1, W2, b2, W3, b3 = weights
    a1 = torch.relu(X @ W1 + b1)
    a2 = torch.relu(a1 @ W2 + b2)
    logits = a2 @ W3 + b3
    probs = torch.softmax(logits, dim=1).detach().numpy()
    preds = np.argmax(probs, axis=1)
    return preds, probs

# 📈 Bootstrap CI
def bootstrap_ci(y_true, y_pred, y_proba, n_boot=1000, alpha=0.95):
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

# 📊 Evaluation
def evaluate_and_export(model_name, signal_name, fold, y_true, y_pred, y_proba, results_dir, prefix):
    os.makedirs(results_dir, exist_ok=True)
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    f1_ci, auc_ci = bootstrap_ci(y_true, y_pred, y_proba)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'roc_auc_ovr': roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr'),
        'report': classification_report(y_true, y_pred, output_dict=True),
        'confusion': confusion_matrix(y_true, y_pred)
    }

    report_df = pd.DataFrame(metrics['report']).transpose()
    report_df.to_csv(f"{results_dir}/{prefix}{signal_name}_{model_name}_report_fold{fold}.csv")
    with open(f"{results_dir}/{prefix}{signal_name}_{model_name}_report_fold{fold}.tex", 'w') as f:
        f.write(report_df.to_latex(float_format="%.3f"))

    pd.DataFrame(metrics['confusion']).to_csv(f"{results_dir}/{prefix}{signal_name}_{model_name}_confusion_fold{fold}.csv")

    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc_ovr"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve (Fold {fold}, {signal_name})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{prefix}roc_curve_{signal_name}_fold{fold}.png", dpi=300)
    plt.close()

    summary = {
        'fold': fold,
        'model': model_name,
        'method': signal_name,
        'accuracy': metrics['accuracy'],
        'f1_macro': metrics['f1_macro'],
        'f1_ci_lower': f1_ci[0],
        'f1_ci_upper': f1_ci[1],
        'roc_auc_ovr': metrics['roc_auc_ovr'],
        'auc_ci_lower': auc_ci[0],
        'auc_ci_upper': auc_ci[1]
    }

    classwise = [{
        'fold': fold,
        'model': model_name,
        'method': signal_name,
        'class': int(cls),
        'precision': report_df.loc[cls, 'precision'],
        'recall': report_df.loc[cls, 'recall'],
        'f1': report_df.loc[cls, 'f1-score']
    } for cls in report_df.index if cls.isdigit()]

    return summary, classwise

def train_cnn(X, y, num_classes, epochs=20, lr=1e-3):
    X = X.unsqueeze(1)  # (batch, 1, length)
    conv1 = torch.randn(8, 1, 5, requires_grad=True)
    conv2 = torch.randn(16, 8, 5, requires_grad=True)
    fc = torch.randn(16 * ((X.shape[2] - 8),), num_classes, requires_grad=True)
    bias = torch.zeros(num_classes, requires_grad=True)

    for _ in range(epochs):
        x = torch.conv1d(X, conv1)
        x = torch.relu(x)
        x = torch.conv1d(x, conv2)
        x = torch.relu(x)
        x = x.view(x.shape[0], -1)
        logits = x @ fc + bias
        log_probs = torch.log_softmax(logits, dim=1)
        loss = torch.nn.functional.nll_loss(log_probs, y)
        loss.backward()
        with torch.no_grad():
            for param in [conv1, conv2, fc, bias]:
                param -= lr * param.grad
                param.grad.zero_()
    return [conv1, conv2, fc, bias]

def predict_cnn(X, weights):
    conv1, conv2, fc, bias = weights
    X = X.unsqueeze(1)
    x = torch.conv1d(X, conv1)
    x = torch.relu(x)
    x = torch.conv1d(x, conv2)
    x = torch.relu(x)
    x = x.view(x.shape[0], -1)
    logits = x @ fc + bias
    probs = torch.softmax(logits, dim=1).detach().numpy()
    preds = np.argmax(probs, axis=1)
    return preds, probs

def train_resnet(X, y, num_classes, epochs=20, lr=1e-3):
    X = X.unsqueeze(1)
    conv = torch.randn(16, 1, 3, requires_grad=True)
    fc = torch.randn(16 * X.shape[2], num_classes, requires_grad=True)
    bias = torch.zeros(num_classes, requires_grad=True)

    for _ in range(epochs):
        x = torch.conv1d(X, conv, padding=1)
        x = torch.relu(x + X.repeat(1, 16, 1))  # residual
        x = x.view(x.shape[0], -1)
        logits = x @ fc + bias
        log_probs = torch.log_softmax(logits, dim=1)
        loss = torch.nn.functional.nll_loss(log_probs, y)
        loss.backward()
        with torch.no_grad():
            for param in [conv, fc, bias]:
                param -= lr * param.grad
                param.grad.zero_()
    return [conv, fc, bias]

def predict_resnet(X, weights):
    conv, fc, bias = weights
    X = X.unsqueeze(1)
    x = torch.conv1d(X, conv, padding=1)
    x = torch.relu(x + X.repeat(1, 16, 1))
    x = x.view(x.shape[0], -1)
    logits = x @ fc + bias
    probs = torch.softmax(logits, dim=1).detach().numpy()
    preds = np.argmax(probs, axis=1)
    return preds, probs



# 🏁 Benchmark
def run_dl_benchmark(data_path, results_dir="results", prefix="classification_dl_"):
    df = pd.read_csv(data_path)    
    gkf = GroupKFold(n_splits=5)
    all_summary, all_classwise = [], []

    for signal_col in ['template_ppg_norm', 'sd_template_ppg_norm']:
        print(f"\n📈 Signal: {signal_col}")
        df_signal = df.dropna(subset=['app_id', 'class_label', signal_col])
        X_all = load_signal(df_signal, signal_col)
        y_all = pd.factorize(df_signal['class_label'])[0]
        groups = df_signal['app_id'].values

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups)):
            print(f"🔁 Fold {fold+1}")
            X_train = torch.tensor(X_all[train_idx], dtype=torch.float32)
            y_train = torch.tensor(y_all[train_idx], dtype=torch.long)
            X_test = torch.tensor(X_all[test_idx], dtype=torch.float32)
            y_test = y_all[test_idx]

            input_dim = X_train.shape[1]
            num_classes = len(np.unique(y_all))
        
            for model_name, train_fn, predict_fn in [
                ("MLP", lambda X, y: train_mlp(X, y, input_dim, num_classes), predict_mlp),
                ("CNN", lambda X, y: train_cnn(X, y, num_classes), predict_cnn),
                ("ResNet", lambda X, y: train_resnet(X, y, num_classes), predict_resnet)
            ]:
                weights = train_fn(X_train, y_train)
                y_pred, y_proba = predict_fn(X_test, weights)
                summary, classwise = evaluate_and_export(model_name, signal_col, fold, y_test, y_pred, y_proba, results_dir, prefix)
                all_summary.append(summary)
                all_classwise.extend(classwise)

    pd.DataFrame(all_summary).to_csv(f"{results_dir}/{prefix}summary_all.csv", index=False)
    pd.DataFrame(all_classwise).to_csv(f"{results_dir}/{prefix}summary_classwise.csv", index=False)

    with open(f"{results_dir}/environment_dl.txt", 'w') as f:
        f.write(f"Python: {sys.version}\n")
        f.write(f"torch: {torch.__version__}\n")
        f.write(f"pandas: {pd.__version__}\n")
        f.write(f"numpy: {np.__version__}\n")

# 📊 Statistička poređenja
def statistical_comparison(summary_csv, output_txt):
    df = pd.read_csv(summary_csv)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("Statistical comparison of F1-macro scores\n\n")
        for method in df['method'].unique():
            subset = df[df['method'] == method]
            models = subset['model'].unique()
            for i, a in enumerate(models):
                for b in models[i+1:]:
                    f1_a = subset[subset['model'] == a]['f1_macro'].values
                    f1_b = subset[subset['model'] == b]['f1_macro'].values
                    if len(f1_a) == len(f1_b) and len(f1_a) > 1:
                        t_stat, t_pval = ttest_rel(f1_a, f1_b)
                        w_stat, w_pval = wilcoxon(f1_a, f1_b)
                        f.write(f"{method}: {a} vs {b}\n")
                        f.write(f"Paired t-test: t = {t_stat:.3f}, p = {t_pval:.5f}\n")
                        f.write(f"Wilcoxon test: W = {w_stat:.3f}, p = {w_pval:.5f}\n")
                        f.write("→ t-test: " + ("Significant difference.\n" if t_pval < 0.05 else "No significant difference.\n"))
                        f.write("→ Wilcoxon: " + ("Significant difference.\n" if w_pval < 0.05 else "No significant difference.\n"))
                        f.write("\n")

if __name__ == "__main__":
    DATA_PATH = "data/train_rws_ppg_classification_dataset.csv"
    RESULTS_DIR = "results"
    PREFIX = "classification_dl_"

    run_dl_benchmark(DATA_PATH, RESULTS_DIR, PREFIX)
    statistical_comparison(f"{RESULTS_DIR}/{PREFIX}summary_all.csv",
                           f"{RESULTS_DIR}/{PREFIX}statistical_significance.txt")