import ast
from itertools import cycle
import os
import re
import numpy as np
from scipy import signal
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef, roc_curve, auc, precision_recall_curve, precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from keras.utils import plot_model
import itertools
import shap

import tensorflow as tf

def str2arr(data_str):
    cleaned_str = re.sub(r'[\[\]\s]', '', data_str)
    return np.array([float(x) for x in cleaned_str.split(',') if x])

def array2str(arr):    
    return ','.join(map(str, arr.flatten()))

def normalize_0_1(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def normalize_m1_1(ppg_signal):
    ppg_signal = np.array(ppg_signal, dtype=np.float32)
    min_val, max_val = np.min(ppg_signal), np.max(ppg_signal)
    return 2 * (ppg_signal - min_val) / (max_val - min_val) - 1  # Normalizacija u opseg [-1, 1]

def sd3(ppg_data):
    ppg_sd = np.zeros(len(ppg_data))
    for i in range(1, len(ppg_data)-1):
        ppg_sd[i] = ppg_data[i-1] - 2*ppg_data[i] + ppg_data[i+1]
    return ppg_sd

def sd5(ppg_data):
    ppg_sd = np.zeros(len(ppg_data))
    for i in range(3, len(ppg_data)-3):
        ppg_sd[i] = -ppg_data[i-2] + 16*ppg_data[i-1] - 30*ppg_data[i] + 16*ppg_data[i+1] - ppg_data[i+2]
    return ppg_sd

def nfFilt(fc, fs, ord, inSig):
    b, a = signal.butter(ord, fc/(fs/2), 'low')
    return signal.filtfilt(b, a, inSig)

def process_ppg(ppg_str, fc=13, fs=100, ord=5, sd_order=-1):
    ppg_array = str2arr(ppg_str) # np.array(ast.literal_eval(ppg_str))
    filtered_ppg = nfFilt(fc, fs, ord, ppg_array)
    if sd_order == 3:
        filtered_ppg = sd3(filtered_ppg)    
    elif sd_order == 5:
        filtered_ppg = sd5(filtered_ppg)        
    return filtered_ppg


def ppg_template_range(ppg, peakInd=35, hr=60, fs=100):
    
    sigMin = np.inf
    signalStartInd = None
    i = peakInd-1    
    while(i>2):
        rightSlope = -3*ppg[i] + 4*ppg[i+1] - ppg[i+2]
        leftSlope = -3*ppg[i-2] + 4*ppg[i-1] - ppg[i]
        if rightSlope>0 and leftSlope<0 and ppg[i]<=sigMin or rightSlope>0 and leftSlope>0 and ppg[i]<=sigMin:
            signalStartInd = i
            sigMin = ppg[i]
        i-=1
    if signalStartInd is None:
        signalStartInd = round(peakInd - .22*60*fs/hr)
    else:
        for i in range(signalStartInd-5, signalStartInd+5):
            if i>1 and ppg[i]<=ppg[i-1] and ppg[i]<=ppg[i+1] and ppg[i]<sigMin:
                sigMin = ppg[i]
                signalStartInd = i
    
    # samples_per_beat = 60*fs / hr 

    sigMin = np.inf
    signalEndInd = None
    i = peakInd+29
    maxSearchInd = peakInd + .9*60*fs/hr
    while(i<maxSearchInd and i<len(ppg)-2):
        rightSlope = -3*ppg[i] + 4*ppg[i+1] - ppg[i+2]
        leftSlope = -3*ppg[i-2] + 4*ppg[i-1] - ppg[i]
        if rightSlope<0 and leftSlope>0 and ppg[i]<=sigMin or rightSlope<0 and leftSlope<0 and ppg[i]<=sigMin:
            signalEndInd = i
            sigMin = ppg[i]
        i+=1
    if signalEndInd is None:
        signalEndInd = round(peakInd + .78*60*fs/hr)
    else:
        for i in range(signalEndInd-5, signalEndInd+5):
            if i<len(ppg)-1 and ppg[i]<=ppg[i-1] and ppg[i]<=ppg[i+1] and ppg[i]<sigMin:
                sigMin = ppg[i]
                signalEndInd = i

    return signalStartInd, signalEndInd

def detect_ab_sdppg(ppg, systolic_peak_index, ppg_start_index):
    ppg = np.asarray(ppg)
    
    # Drugi i treći izvod
    sdppg = np.zeros(len(ppg))
    for i in range(1, len(ppg) - 1):
        sdppg[i] = ppg[i - 1] - 2 * ppg[i] + ppg[i + 1]

    d3ppg = np.zeros(len(ppg))
    for i in range(2, len(ppg) - 2):
        d3ppg[i] = ppg[i - 2] - 2 * ppg[i - 1] + 2 * ppg[i + 1] - ppg[i + 2]

    # Pik A: maksimum SDPPG između početka i sistolnog pika
    a_value = -np.inf
    a_index = None
    for i in range(ppg_start_index + 1, systolic_peak_index - 3):
        if sdppg[i] >= sdppg[i - 1] and sdppg[i] >= sdppg[i + 1] and sdppg[i] >= a_value:
            a_value = sdppg[i]
            a_index = i

    if a_index is None:
        for i in range(ppg_start_index + 1, systolic_peak_index - 3):
            if d3ppg[i - 1] < 0 and d3ppg[i] >= 0 or d3ppg[i - 1] <= 0 and d3ppg[i] > 0:
                a_index = i
                a_value = sdppg[i]

    if a_index is None:
        return np.nan, np.nan, np.nan, np.nan
    
    # Pik B: prvi nulti presek trećeg izvoda nakon A, pre sistolnog pika
    b_index = None
    b_value = np.inf
    for i in range(a_index + 1, systolic_peak_index-3):
        if sdppg[i]<=sdppg[i-1] and sdppg[i]<=sdppg[i+1] and sdppg[i]<=b_value:
            b_index = i
            b_value = sdppg[i]
            
    if b_index is None:
        for i in range(a_index + 1, systolic_peak_index-3):
            if d3ppg[i - 1] > 0 and d3ppg[i] <= 0 or d3ppg[i - 1] >= 0 and d3ppg[i] < 0:
                b_index = i
                b_value = sdppg[i]
    
    if b_index is None:
        d4 = np.zeros_like(ppg)
        for i in range(2, len(ppg) - 2):
            d4[i] = 1000*(ppg[i - 2] - 4 * ppg[i - 1] + 6 * ppg[i]
                        - 4 * ppg[i + 1] + ppg[i + 2])
        d4 = nfFilt(fc=10, fs=100, ord=5, inSig=d4)
        d4LocMax = -np.inf
        for i in range(a_index + 1, systolic_peak_index-3):
            if d4[i - 1] <= d4[i] and d4[i + 1] <= d4[i] or d3ppg[i - 1] >= 0 and d4[i] >= d4LocMax:
                b_index = i
                b_value = sdppg[i]
                d4LocMax = d4[i]
        
    if b_index is None:
        b_index = np.nan
        b_value = np.nan

    return a_index, a_value, b_index, b_value



def sum_of_2_gaussians(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    gauss1 = amp1 * np.exp(-(x - mean1)**2 / (2 * sigma1**2))
    gauss2 = amp2 * np.exp(-(x - mean2)**2 / (2 * sigma2**2))
    return gauss1 + gauss2

def sum_of_3_gaussians(x, amp1, mean1, sigma1, amp2, mean2, sigma2, amp3, mean3, sigma3):
    gauss1 = amp1 * np.exp(-(x - mean1)**2 / (2 * sigma1**2))
    gauss2 = amp2 * np.exp(-(x - mean2)**2 / (2 * sigma2**2))
    gauss3 = amp3 * np.exp(-(x - mean3)**2 / (2 * sigma3**2))
    return gauss1 + gauss2 + gauss3

# Definicija analitičkog Jakobijana za fitovanje
def jacobian_2g(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    gauss1 = np.exp(-(x - mean1)**2 / (2 * sigma1**2))
    gauss2 = np.exp(-(x - mean2)**2 / (2 * sigma2**2))

    d_amp1 = gauss1
    d_mean1 = amp1 * gauss1 * (x - mean1) / sigma1**2
    d_sigma1 = amp1 * gauss1 * (x - mean1)**2 / sigma1**3

    d_amp2 = gauss2
    d_mean2 = amp2 * gauss2 * (x - mean2) / sigma2**2
    d_sigma2 = amp2 * gauss2 * (x - mean2)**2 / sigma2**3

    return np.vstack([d_amp1, d_mean1, d_sigma1, d_amp2, d_mean2, d_sigma2]).T

def jacobian_3g(x, amp1, mean1, sigma1, amp2, mean2, sigma2, amp3, mean3, sigma3):
    gauss1 = np.exp(-(x - mean1)**2 / (2 * sigma1**2))
    gauss2 = np.exp(-(x - mean2)**2 / (2 * sigma2**2))
    gauss3 = np.exp(-(x - mean3)**2 / (2 * sigma3**2))

    d_amp1 = gauss1
    d_mean1 = amp1 * gauss1 * (x - mean1) / sigma1**2
    d_sigma1 = amp1 * gauss1 * (x - mean1)**2 / sigma1**3

    d_amp2 = gauss2
    d_mean2 = amp2 * gauss2 * (x - mean2) / sigma2**2
    d_sigma2 = amp2 * gauss2 * (x - mean2)**2 / sigma2**3

    d_amp3 = gauss3
    d_mean3 = amp3 * gauss3 * (x - mean3) / sigma3**2
    d_sigma3 = amp3 * gauss3 * (x - mean3)**2 / sigma3**3

    return np.vstack([d_amp1, d_mean1, d_sigma1, d_amp2, d_mean2, d_sigma2, d_amp3, d_mean3, d_sigma3]).T

def specificity_score(y_true, y_pred):
    """Izračunava specifičnost za svaku klasu."""
    cm = confusion_matrix(y_true, y_pred)
    spec = []
    for i in range(len(cm)):
        TN = cm.sum() - cm[i,:].sum() - cm[:,i].sum() + cm[i,i]
        FP = cm[:,i].sum() - cm[i,i]
        spec.append(TN / (TN + FP) if (TN + FP) != 0 else 0.0)
    return np.array(spec)


def classification_report_with_specificity(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    
    # Formatiranje izveštaja
    headers = ["Class", "Precision", "Recall", "F1", "Specificity", "Support"]
    rows = zip(np.unique(y_true), precision, recall, f1, spec, support)
    
    print("\nClassification report:")
    print("-"*80)
    print(f"{headers[0]:<10} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]}")
    print("-"*80)
    for row in rows:
        print(f"{row[0]:<10} {row[1]:<12.2f} {row[2]:<12.2f} {row[3]:<12.2f} {row[4]:<12.2f} {row[5]}")
    
    # Proseci
    avg_precision = np.average(precision, weights=support)
    avg_recall = np.average(recall, weights=support)
    avg_f1 = np.average(f1, weights=support)
    avg_spec = np.average(spec, weights=support)
    total_support = np.sum(support)
    
    print("\nSummary:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"Macro avg - Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f}, Specificity: {avg_spec:.2f}")
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Matthews Correlation Coefficient:{mcc:.2f}")


def plot_confusion_matrix(y_true, y_pred, classes=['Class 1', 'Class 2', 'Class 3', 'Class 4']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Classes') 
    plt.ylabel('True Classes')
    plt.tight_layout()
    plt.show()

def plot_roc_auc_ovr(model, X_test, y_test, classesTarg=None, y_score_res=None):
    """
    Prikazuje ROC krive i AUC vrednosti za svaku klasu koristeći One-vs-Rest pristup
    
    Args:
        model: Obučeni klasifikacioni model
        X_test: Testni skup podataka
        y_test: Stvarne klase testnog skupa
        target_cols: Lista naziva ciljnih kolona (['class1', 'class2', 'class3', 'class4'])
    """
    # Binarizacija izlaza
    if classesTarg is not None:
        y_test_bin = label_binarize(y_test, classes=classesTarg)
    else:
        y_test_bin = label_binarize(y_test, classes=model.classes_)

    n_classes = y_test_bin.shape[1]
    
    # Predikcija verovatnoća
    if y_score_res is not None:
        y_score = y_score_res
    else:
        # Provera da li model ima metodu predict_proba, predict ili decision_function
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "predict"):
            y_score = model.predict(X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            raise AttributeError("Model nema ni predict_proba ni predict metodu ni decision_function metodu.")

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "predict"):
        y_score = model.predict(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise AttributeError("Model nema ni predict_proba ni predict metodu ni decision_function metodu.")
    
    # Podešavanje izgleda grafikona
    # Podešavanje izgleda grafikona
    plt.figure(figsize=(10, 8))
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:n_classes]
    
    # Računanje ROC krive i AUC za svaku klasu
    for i in range(n_classes):
        # Provera da li postoji bar jedan pozitivan primer u test skupu
        if np.sum(y_test_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            
            class_label = classesTarg[i] if classesTarg is not None else f"Class {i+1}"
            plt.plot(fpr, tpr, color=colors[i], 
                    label=f'ROC {class_label} (AUC = {roc_auc:.2f})')
        else:
            print(f"Upozorenje: Nema pozitivnih primera za klasu {i} u test skupu. Preskačem ROC krivu.")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve - One-vs-Rest')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_roc_auc_ovr_gru(model, X_test, y_test, target_cols=None, y_score_res=None):
    """
    Prikazuje ROC krive za sve klase koristeći One-vs-Rest pristup
    
    Args:
        model: Obučeni Keras model
        X_test: Testni skup podataka
        y_test: Stvarne klase (one-hot encoded)
        target_cols: Lista naziva klasa
        y_score_res: Već izračunate predikcije (opciono)
    """
    # Ako y_test nije one-hot, pretvorimo ga
    if len(y_test.shape) == 1 or y_test.shape[1] == 1:
        y_test = label_binarize(y_test, classes=np.arange(len(target_cols)))
    
    n_classes = y_test.shape[1]
    
    # Ako nisu prosleđeni nazivi klasa, koristimo generičke
    if target_cols is None:
        target_cols = [f'Class {i}' for i in range(n_classes)]
    
    # Dobijanje predikcija ako nisu već prosleđene
    if y_score_res is None:
        y_score = model.predict(X_test)
    else:
        y_score = y_score_res
    
    # Podešavanje izgleda grafikona
    plt.figure(figsize=(10, 8))
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:n_classes]
    
    # Računanje ROC krive i AUC za svaku klasu
    for i in range(n_classes):
        # Provera da li postoji bar jedan pozitivan primer u test skupu
        if np.sum(y_test[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i],
                    label=f'{target_cols[i]} (AUC = {roc_auc:.2f})')
        else:
            print(f"Upozorenje: Nema pozitivnih primera za klasu {target_cols[i]} u test skupu. Preskačem ROC krivu.")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pr_auc(model, X_test, y_test, classesTarg=None):
    
    if classesTarg is not None:
        y_test_bin = label_binarize(y_test, classes=classesTarg)
    else:
        y_test_bin = label_binarize(y_test, classes=model.classes_)

    n_classes = y_test_bin.shape[1]

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "predict"):
        y_score = model.predict(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise AttributeError("Model nema ni predict_proba ni predict metodu ni decision_function metodu.")
    # y_score = model.predict_proba(X_test)
    
    plt.figure(figsize=(8, 6))

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:n_classes]
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color=colors[i], 
                label=f'{classesTarg[i] if classesTarg is not None else model.classes_[i]} (AUC = {pr_auc:.2f})')
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.grid()
    plt.show()

def plot_pr_auc_gru(model, X_test, y_test, target_cols=None, y_score_res=None):
    """
    Prikazuje Precision-Recall krive za sve klase za GRU model
    
    Args:
        model: Obučeni Keras GRU model
        X_test: Testni skup podataka (shape: [samples, timesteps, features])
        y_test: Stvarne klase (one-hot encoded ili indeksi klasa)
        target_cols: Lista naziva klasa (npr. ['Class1', 'Class2', 'Class3', 'Class4'])
        y_score_res: Već izračunate predikcije (opciono)
    """
    # Provera formata y_test
    if len(y_test.shape) == 1 or y_test.shape[1] == 1:
        # Ako y_test nije one-hot, pretvorimo ga
        classes = np.arange(len(target_cols)) if target_cols is not None else np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
    else:
        y_test_bin = y_test
    
    n_classes = y_test_bin.shape[1]
    
    # Postavljanje naziva klasa ako nisu prosleđeni
    if target_cols is None:
        target_cols = [f'Class {i}' for i in range(n_classes)]
    
    # Dobijanje predikcija ako nisu već prosleđene
    if y_score_res is None:
        y_score = model.predict(X_test)
    else:
        y_score = y_score_res
    
    # Podešavanje izgleda grafikona
    plt.figure(figsize=(10, 8))
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:n_classes]
    
    # Računanje PR krive za svaku klasu
    for i in range(n_classes):
        # Provera da li postoji bar jedan pozitivan primer u test skupu
        if np.sum(y_test_bin[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, color=colors[i],
                    label=f'{target_cols[i]} (AUC = {pr_auc:.2f})',
                    linewidth=2)
        else:
            print(f"Upozorenje: Nema pozitivnih primera za klasu {target_cols[i]} u test skupu. Preskačem PR krivu.")
    
    # Podešavanje grafikona
    plt.plot([0, 1], [np.sum(y_test_bin) / len(y_test_bin)] * 2, 'k--', label='Random Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (One-vs-Rest)', fontsize=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Dodatna informacija o balansiranosti klasa
    class_dist = np.sum(y_test_bin, axis=0)
    plt.text(1.05, 0.2, 
             'Class Distribution:\n' + '\n'.join([f'{target_cols[i]}: {class_dist[i]}' for i in range(n_classes)]),
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()

# Funkcija za izračunavanje MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_training_history(history):
    print("\nTraining History plot:")
    for epoch, (loss, val_loss, acc, val_acc) in enumerate(zip(history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy'])):
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val Loss={val_loss:.4f}, Accuracy={acc:.4f}, Val Accuracy={val_acc:.4f}")
    
    plt.figure(figsize=(12, 5))

    # Gubitak kroz epohe
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# def plot_regr_training_history(history):
    
#     try:
#         print("\nRegression Training History plot:")
#         for epoch, (loss, val_loss, mae, val_mae) in enumerate(zip(
#             history.history['loss'], history.history['val_loss'], 
#             history.history['mae'], history.history['val_mae'])):
#             print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val Loss={val_loss:.4f}, MAE={mae:.4f}, Val MAE={val_mae:.4f}")
#     except Exception as e:
#         print(f"Error: Available keys: {history.history.keys()}")
#         print(f"Error: {e}")

#     import matplotlib.pyplot as plt

#     # Kreiranje podgrafikona
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 red, 2 kolone

#     # Prikaz gubitka (Loss)
#     axes[0].plot(history.history['loss'], label='Loss')
#     axes[0].plot(history.history['val_loss'], label='Validation Loss')
#     axes[0].set_xlabel('Epochs')
#     axes[0].set_ylabel('Loss (MAE)')
#     axes[0].legend()
#     axes[0].set_title('Loss History')

#     # Prikaz MAE
#     axes[1].plot(history.history['mae'], label='Train MAE')
#     axes[1].plot(history.history['val_mae'], label='Validation MAE')
#     axes[1].set_xlabel('Epochs')
#     axes[1].set_ylabel('Mean Absolute Error (MAE)')
#     axes[1].legend()
#     axes[1].set_title('MAE History')

#     plt.tight_layout()  # Poboljšava raspored grafikona
#     plt.show()
def plot_regr_training_history(history):
    import matplotlib.pyplot as plt

    print("\n📈 Regression Training History plot:")
    print(f"Available keys: {list(history.history.keys())}")

    # Ispis vrednosti po epohama ako postoje odgovarajući ključevi
    try:
        epochs = len(history.history['loss'])
        for epoch in range(epochs):
            log_line = f"Epoch {epoch+1}:"
            for key in ['loss', 'val_loss', 'mae', 'val_mae']:
                if key in history.history:
                    log_line += f" {key}={history.history[key][epoch]:.4f}"
            print(log_line)
    except Exception as e:
        print(f"⚠️ Error printing per-epoch metrics: {e}")

    # Kreiranje podgrafikona dinamički na osnovu dostupnih ključeva
    keys = history.history.keys()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss subplot
    axes[0].set_title('Loss History')
    if 'loss' in keys:
        axes[0].plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in keys:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # MAE subplot (ako postoji)
    axes[1].set_title('MAE History')
    mae_plotted = False
    if 'mae' in keys:
        axes[1].plot(history.history['mae'], label='Train MAE')
        mae_plotted = True
    if 'val_mae' in keys:
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
        mae_plotted = True

    if mae_plotted:
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'MAE not available', ha='center', va='center')
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def evaluate_classification_model(model, X_test, y_test, target_cols=['Class 1', 'Class 2', 'Class 3', 'Class 4'], y_pred_probs=None):
    print("\nEvaluating model...")

    if hasattr(model, "summary"):
        print(f"\nModel Summary:{model.summary()}")
    
    # plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True, show_layer_activations=True, show_dtype=True, show_trainable=True, to_file='model_cnn.png')
    
    y_pred = model.predict(X_test)
    if len(y_pred.shape) > 1 and y_pred.shape[1] == len(target_cols):
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred.astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred_classes):.2f}")

    classification_report_with_specificity(y_test, y_pred_classes)

    plot_confusion_matrix(y_test, y_pred_classes, target_cols)

    y_pred_one_hot = np.eye(len(target_cols))[y_pred_classes]

    # plot_roc_auc_ovr_gru(model, X_test, y_test, target_cols=target_cols, y_score_res=(y_pred_probs if y_pred_probs is not None else y_pred) )
    # plot_pr_auc_gru(model, X_test, y_test, target_cols=target_cols, y_score_res=(y_pred_probs if y_pred_probs is not None else y_pred))
    plot_multiclass_roc_pr_curves(y_test, y_pred_probs if y_pred_probs is not None else y_pred, class_names=['Class 1', 'Class 2', 'Class 3', 'Class 4'], figsize=(12, 5))
    plot_multiclass_ovo_roc_pr_curves(y_test, y_pred_probs if y_pred_probs is not None else y_pred, class_names=['Class 1', 'Class 2', 'Class 3', 'Class 4'], figsize=(12, 5))
    plot_few_classifications(model, X_test, y_test, cntPerClass=5)


def plot_multiclass_roc_pr_curves(y_true, y_scores, class_names=['Class 1', 'Class 2', 'Class 3', 'Class 4'], figsize=(12, 5)):
    
    if y_true.ndim == 1:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    else:
        y_true_bin = y_true
    
    n_classes = y_true_bin.shape[1]
    
    # Set class names if not provided
    if class_names is None:
        class_names = [f'Class {i+1}' for i in range(n_classes)]
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute Precision-Recall curve and average precision for each class
    precision, recall, avg_precision = {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    
    # Compute micro-average PR curve and average precision
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
    avg_precision["micro"] = average_precision_score(y_true_bin, y_scores, average="micro")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot ROC curves
    # colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    #                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown'][:n_classes]
    
    for i, color in zip(range(n_classes), colors):
        ax1.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:0.2f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Multiclass ROC Curve')
    ax1.legend(loc="lower right")
    
    # Plot PR curves
    for i, color in zip(range(n_classes), colors):
        ax2.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{class_names[i]} (AP = {avg_precision[i]:0.2f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Multiclass Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    plt.show()
    return fig

def plot_multiclass_ovo_roc_pr_curves(y_true, y_scores, class_names=None, figsize=(12, 6)):
    """
    Plot ROC and Precision-Recall curves for One-vs-One multiclass classification.

    y_true: True class labels (1D array).
    y_scores: Predicted probability scores (2D array).
    class_names: List of class names (optional).
    figsize: Size of the output figure.
    """
    # Ako su klase u string formatu, konvertuj u numeričke vrednosti
    if isinstance(y_true[0], str):
        class_map = {name: i for i, name in enumerate(["class1", "class2", "class3", "class4"])}
        y_true = np.array([class_map[label] for label in y_true])

    # Binarizacija y_true ako nije već one-hot encoded
    unique_classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=unique_classes) if y_true.ndim == 1 else y_true
    n_classes = y_true_bin.shape[1]

    # Postavljanje naziva klasa ako nisu definisane
    if class_names is None:
        class_names = [f'Class {i+1}' for i in range(n_classes)]

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'gray', 'yellow'][:n_classes * (n_classes - 1) // 2]
    
    # Kreiranje kombinacija klasa za OvO ROC i PR analizu
    class_pairs = list(itertools.combinations(range(n_classes), 2))

    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, avg_precision = {}, {}, {}

    for (class1, class2) in class_pairs:
        if class1 not in unique_classes or class2 not in unique_classes:
            print(f"Skipping {class_names[class1]} vs {class_names[class2]} due to missing class")
            continue

        # Filter podataka za samo dve klase
        mask = (y_true == class1) | (y_true == class2)
        y_true_filtered = y_true[mask]
        y_scores_filtered = y_scores[mask, class1]  # Ispravljeno indeksiranje
        
        y_true_bin_filtered = np.where(y_true_filtered == class1, 1, 0)

        unique_vals = np.unique(y_true_bin_filtered)
        if len(unique_vals) < 2:
            print(f"Skipping {class_names[class1]} vs {class_names[class2]} due to uniform labels: {unique_vals}")
            continue

        # Računanje ROC i PR vrednosti
        fpr[(class1, class2)], tpr[(class1, class2)], _ = roc_curve(y_true_bin_filtered, y_scores_filtered, pos_label=1)
        roc_auc[(class1, class2)] = auc(fpr[(class1, class2)], tpr[(class1, class2)])

        precision[(class1, class2)], recall[(class1, class2)], _ = precision_recall_curve(y_true_bin_filtered, y_scores_filtered)
        avg_precision[(class1, class2)] = average_precision_score(y_true_bin_filtered, y_scores_filtered)

    # Ispis rezultata u CSV formatu
    print("\Class Combination,ROC AUC,PR AP")
    for (class1, class2) in class_pairs:
        if (class1, class2) in roc_auc:
            print(f"{class_names[class1]} vs {class_names[class2]},{roc_auc[(class1, class2)]:.5f},{avg_precision[(class1, class2)]:.5f}")

    # Vizualizacija ROC i PR krivih
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ROC krive
    for (idx, (class1, class2)) in enumerate(class_pairs):
        if (class1, class2) in roc_auc:
            ax1.plot(fpr[(class1, class2)], tpr[(class1, class2)], color=colors[idx % len(colors)],
                     label=f"ROC {class_names[class1]} vs {class_names[class2]} (AUC = {roc_auc[(class1, class2)]:.3f})")

    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("OvO ROC Curve")
    ax1.legend()
    ax1.grid()

    # Precision-Recall krive
    for (idx, (class1, class2)) in enumerate(class_pairs):
        if (class1, class2) in avg_precision:
            ax2.plot(recall[(class1, class2)], precision[(class1, class2)], color=colors[idx % len(colors)],
                     label=f"PR {class_names[class1]} vs {class_names[class2]} (AP = {avg_precision[(class1, class2)]:.3f})")

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("OvO Precision-Recall Curve")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()

    return fig

def _plot_multiclass_ovo_roc_pr_curves(y_true, y_scores, class_names=None, figsize=(12, 6)):
    """
    Plot ROC and Precision-Recall curves for One-vs-One multiclass classification.

    y_true: True class labels (1D array).
    y_scores: Predicted probability scores (2D array).
    class_names: List of class names (optional).
    figsize: Size of the output figure.
    """

    # Provera distribucije klasa
    unique_classes, counts = np.unique(y_true, return_counts=True)
    print("Distribucija klasa:", dict(zip(unique_classes, counts)))

    # Binarizacija y_true ako nije već one-hot encoded
    y_true_bin = label_binarize(y_true, classes=unique_classes) if y_true.ndim == 1 else y_true
    n_classes = y_true_bin.shape[1]

    # Postavljanje naziva klasa ako nisu definisane
    if class_names is None:
        class_names = [f'Class {i+1}' for i in range(n_classes)]

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'gray', 'yellow'][:n_classes * (n_classes - 1) // 2]
    
    # Kreiranje kombinacija klasa za OvO ROC i PR analizu
    class_pairs = list(itertools.combinations(range(n_classes), 2))

    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, avg_precision = {}, {}, {}

    for (class1, class2) in class_pairs:
        if class1 not in unique_classes or class2 not in unique_classes:
            print(f"Skipping {class_names[class1]} vs {class_names[class2]} due to missing class")
            continue

        # Filter podataka za samo dve klase
        mask = (y_true == class1) | (y_true == class2)
        y_true_filtered = y_true[mask]
        y_scores_filtered = y_scores[mask][:, class1]  # Uzima samo verovatnoću za class1
        
        # Ispravna binarizacija
        y_true_bin_filtered = np.where(y_true_filtered == class1, 1, 0)

        # Provera validnosti podataka
        if len(np.unique(y_true_bin_filtered)) < 2:
            print(f"Skipping {class_names[class1]} vs {class_names[class2]} due to uniform labels ({np.unique(y_true_bin_filtered)})")
            continue

        # Računanje ROC i PR vrednosti
        fpr[(class1, class2)], tpr[(class1, class2)], _ = roc_curve(y_true_bin_filtered, y_scores_filtered, pos_label=1)
        roc_auc[(class1, class2)] = auc(fpr[(class1, class2)], tpr[(class1, class2)])

        precision[(class1, class2)], recall[(class1, class2)], _ = precision_recall_curve(y_true_bin_filtered, y_scores_filtered)
        avg_precision[(class1, class2)] = average_precision_score(y_true_bin_filtered, y_scores_filtered)

    # Ispis rezultata u CSV formatu
    print("\nClass Combination,ROC AUC,PR AP")
    for (class1, class2) in class_pairs:
        if (class1, class2) in roc_auc:
            print(f"{class_names[class1]} vs {class_names[class2]},{roc_auc[(class1, class2)]:.5f},{avg_precision[(class1, class2)]:.2f}")

    # Vizualizacija ROC i PR krivih
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ROC krive
    for (idx, (class1, class2)) in enumerate(class_pairs):
        if (class1, class2) in roc_auc:
            ax1.plot(fpr[(class1, class2)], tpr[(class1, class2)], color=colors[idx % len(colors)],
                     label=f"ROC {class_names[class1]} vs {class_names[class2]} (AUC = {roc_auc[(class1, class2)]:.2f})")

    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("OvO ROC Curve")
    ax1.legend()
    ax1.grid()

    # Precision-Recall krive
    for (idx, (class1, class2)) in enumerate(class_pairs):
        if (class1, class2) in avg_precision:
            ax2.plot(recall[(class1, class2)], precision[(class1, class2)], color=colors[idx % len(colors)],
                     label=f"PR {class_names[class1]} vs {class_names[class2]} (AP = {avg_precision[(class1, class2)]:.2f})")

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("OvO Precision-Recall Curve")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()

    return fig

def bland_altman_with_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    avg = (y_true + y_pred) / 2

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(3, 4)

    # Glavni Bland–Altman plot
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_main.scatter(avg, residuals, alpha=0.6)
    ax_main.axhline(0, color='gray', linestyle='--')
    ax_main.set_xlabel('Mean of Actual and Predicted')
    ax_main.set_ylabel('Residuals')
    ax_main.set_title("Bland–Altman Plot")

    # Rotirana raspodela reziduala (KDE)
    ax_kde = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    sns.kdeplot(y=residuals, ax=ax_kde, fill=True, alpha=0.6, linewidth=1.5)
    ax_kde.axhline(0, color='gray', linestyle='--')
    ax_kde.set_xticks([])
    ax_kde.set_ylabel('')
    ax_kde.set_xlabel('Density')

    plt.tight_layout()
    plt.show()

def integrated_gradients(model, input_tensor, baseline=None, steps=50):
    if baseline is None:
        baseline = tf.zeros_like(input_tensor)  # (1, 70, 1)

    # Interpolacija: dobijaš (steps+1, 70, 1)
    alphas = tf.linspace(0.0, 1.0, steps + 1)[:, tf.newaxis, tf.newaxis]  # (steps+1, 1, 1)
    interpolated_inputs = baseline + alphas * (input_tensor - baseline)   # broadcasting

    # Računanje gradijenata
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs)  # (steps+1, 1)
    grads = tape.gradient(predictions, interpolated_inputs)  # (steps+1, 70, 1)

    # Trapezna integracija
    avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2.0, axis=0)  # (70, 1)
    integrated_grads = (input_tensor - baseline)[0] * avg_grads  # (70, 1)

    return integrated_grads.numpy().squeeze()  # => (70,)

# def visualize_integrated_gradients(model, X_sample, sample_index=0):
#     input_tensor = tf.convert_to_tensor(X_sample[sample_index:sample_index+1], dtype=tf.float32)

#     attributions = integrated_gradients(model, input_tensor)

#     plt.figure(figsize=(12, 4))
#     plt.plot(attributions, label='Integrated Gradients Attribution')
#     plt.title(f'PPG Sample {sample_index} - Feature Contribution to Prediction')
#     plt.xlabel('PPG Sample Index')
#     plt.ylabel('Attribution Value')
#     plt.axhline(0, color='gray', linestyle='--')
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

def visualize_integrated_gradients_sdppg(model, X_sample_sd, sample_index=0, age=None, age_est=None, X_sample_ppg=None, save_path=None):
    """
    Visualizes Integrated Gradients alongside the original PPG signal,
    and displays true age if provided.
    """
    import tensorflow as tf
    import matplotlib.pyplot as plt

    input_tensor = tf.convert_to_tensor(X_sample_sd[sample_index:sample_index+1], dtype=tf.float32)
    attributions = integrated_gradients(model, input_tensor)
    attributions = attributions.squeeze()
    signal = X_sample_sd[sample_index].squeeze()

    fig, ax1 = plt.subplots(figsize=(6, 5))

    # Plot original PPG signal
    ax1.set_title(f"SD PPG vs Feature Attribution (Sample {sample_index})", fontsize=14)
    ax1.plot(signal, 'b-', label='Second derivative PPG')
    ax1.set_xlabel("Sample Index", fontsize=12)
    ax1.set_ylabel("Amplitude", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    if X_sample_ppg is not None:
        # Plot original PPG signal if provided
        ppg_signal = X_sample_ppg
        ax1.plot(ppg_signal, 'g-', label='Original PPG Signal', alpha=0.5)

    # Plot attribution on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(attributions, 'r--', label='Integrated Gradients Attribution')
    ax2.set_ylabel("Attribution", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Legend and optional age annotation
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    if age is not None:
        plt.text(0.55, 0.2, f"True Age: {age:.1f} years",
                 transform=ax1.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')
    if age_est is not None:
        plt.text(0.55, 0.1, f"Estimated Age: {age_est:.1f} years",
                 transform=ax1.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')  

    plt.tight_layout()

    if save_path:
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        filename = f"{save_path}/ig_sdppg_age{age:.1f}_pred{age_est:.1f}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")   
    else:
        plt.show()

def average_integrated_gradients(model, X_samples, steps=50, sample_count=100):
    print("Computing average Integrated Gradients...")

    # Uzimamo nasumične uzorke ako ih ima više od sample_count
    if X_samples.shape[0] > sample_count:
        indices = np.random.choice(X_samples.shape[0], sample_count, replace=False)
        X_subset = X_samples[indices]
    else:
        X_subset = X_samples

    X_subset = tf.convert_to_tensor(X_subset, dtype=tf.float32)

    all_attributions = []

    for i in range(X_subset.shape[0]):
        input_tensor = X_subset[i:i+1]
        baseline = tf.zeros_like(input_tensor)
        interpolated_inputs = tf.stack([
            baseline + (float(step) / steps) * (input_tensor - baseline)
            for step in range(steps + 1)
        ], axis=0)  # Oblik: (steps+1, 70, 1)

        interpolated_inputs = tf.reshape(interpolated_inputs, (steps + 1, input_tensor.shape[1], input_tensor.shape[2]))

        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs)
            preds = model(interpolated_inputs)
        grads = tape.gradient(preds, interpolated_inputs)
        avg_grads = tf.reduce_mean(grads[:-1] + grads[1:], axis=0) / 2.0
        integrated_grad = (input_tensor - baseline) * avg_grads
        all_attributions.append(integrated_grad.numpy())

    mean_attribution = np.mean(np.concatenate(all_attributions, axis=0), axis=0).squeeze()

    # Vizuelizacija
    plt.figure(figsize=(12, 4))
    plt.plot(mean_attribution, label='Prosečna atribucija (Integrated Gradients)')
    plt.title('Srednji uticaj PPG odmeraka na predikciju starosti (IG)')
    plt.xlabel('Indeks odmerka PPG signala')
    plt.ylabel('Atribuciona vrednost')
    plt.axhline(0, color='gray', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mean_attribution    



def visualize_shap_vs_ig(model, X_sample, sample_index=0, age=None, age_est=None):
    """
    Upoređuje SHAP i Integrated Gradients na istom PPG uzorku.
    """
    # Priprema ulaza
    input_tensor = tf.convert_to_tensor(X_sample[sample_index:sample_index+1], dtype=tf.float32)
    signal = X_sample[sample_index].squeeze()

    # --- Integrated Gradients ---
    ig_attributions = integrated_gradients(model, input_tensor)
    ig_attributions = ig_attributions.numpy().squeeze()

    # --- SHAP (DeepExplainer) ---
    background = X_sample[np.random.choice(len(X_sample), size=50, replace=False)]
    explainer = shap.DeepExplainer(model, tf.convert_to_tensor(background, dtype=tf.float32))
    shap_values = explainer.shap_values(tf.convert_to_tensor(X_sample[sample_index:sample_index+1], dtype=tf.float32))
    shap_attributions = shap_values[0].squeeze()

    # --- Vizualizacija ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(signal, 'k', label='PPG Signal')
    axs[0].set_title("Original PPG Signal")
    axs[0].legend()

    axs[1].plot(ig_attributions, 'r', label='Integrated Gradients')
    axs[1].set_title("Integrated Gradients Attributions")
    axs[1].legend()

    axs[2].plot(shap_attributions, 'b', label='SHAP Values')
    axs[2].set_title("SHAP Attributions (DeepExplainer)")
    axs[2].legend()

    if age is not None:
        axs[0].text(0.01, 0.95, f"True Age: {age:.1f}", transform=axs[0].transAxes, fontsize=11)
    if age_est is not None:
        axs[0].text(0.01, 0.85, f"Estimated Age: {age_est:.1f}", transform=axs[0].transAxes, fontsize=11)

    plt.xlabel("Sample Index")
    plt.tight_layout()
    plt.show()

def plot_regression_scatter(y_test, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, c='blue', edgecolors='w', linewidth=0.5)
    plt.plot([10, 90], [10, 90], 'r--')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.title('Actual vs Predicted Age')
    plt.grid(True)
    plt.show()

# def plot_few_classifications(model, X_test, y_test, target_cols=['Class 1', 'Class 2', 'Class 3', 'Class 4'], cntPerClass=4):
#     plt.figure(figsize=(10, 6))
#     for i in range(4):
#         # indices = np.where(np.argmax(y_test, axis=1) == i)[0][:3]
#         indices = np.where(y_test == i)[0][:cntPerClass] 
#         for idx in indices:
#             plt.plot(X_test[idx], label=f'True: {target_cols[i]}, Pred: {target_cols[np.argmax(model.predict(X_test[idx:idx+1]))]}')
#     plt.title('Sample PPG Signals with Predictions')
#     plt.xlabel('Time')
#     plt.ylabel('Normalized Amplitude')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()

def plot_few_classifications(model, X_test, y_test, target_cols=['Class 1', 'Class 2', 'Class 3', 'Class 4'], cntPerClass=4):
    plt.figure(figsize=(12, 8))
    
    # Definišite paletu boja za svaku klasu
    class_colors = ['red', 'green', 'blue', 'purple']  # Možete dodati više boja po potrebi
    
    for i in range(len(target_cols)):
        # indices = np.where(y_test == i)[0][:cntPerClass]
        all_indices = np.where(y_test == i)[0]
        indices = np.random.choice(all_indices, size=min(cntPerClass, len(all_indices)), replace=False)
        
        for j, idx in enumerate(indices):
            # Predviđanje modela
            pred = np.argmax(model.predict(X_test[idx:idx+1], verbose=0))
            
            # Generisanje jedinstvene nijanse za svaki primer unutar klase
            color = class_colors[i]
            alpha = 0.7 + 0.3 * (j / cntPerClass)  # Varijacija providnosti
            linewidth = 1.5 + 0.5 * j  # Varijacija debljine linije
            
            plt.plot(X_test[idx], 
                     color=color,
                     alpha=alpha,
                     linewidth=linewidth,
                     linestyle=['-', '--', ':', '-.'][j % 4],  # Različiti stilovi linija
                     label=f'True: {target_cols[i]}, Pred: {target_cols[pred]}')
    
    plt.title('Sample PPG Signals with Predictions')
    plt.xlabel('Time')
    plt.ylabel('Normalized Amplitude')
    
    # Poboljšana legenda
    plt.legend(bbox_to_anchor=(1.05, 1), 
              loc='upper left',
              fontsize='small',
              ncol=1,
              framealpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate_power_penalty(num_classes=4, alpha=1.0, k=1):
    """
    Generates a penalty matrix where penalties increase based on the distance between classes.
    W[i,j] = 1 + α*∣i−j∣^k 

    Args:
        num_classes (int): Number of classes.
        alpha (float): Penalty strength multiplier. Default: 1.0.
        mode (str): "linear" or "quadratic" penalty progression. Default: "linear".

    Returns:
        np.ndarray: Penalty matrix of shape (num_classes, num_classes).

    Example:
        >>> generate_power_penalty(4, alpha=1.0, k=1)
        [[1., 2., 3., 4.],
         [2., 1., 2., 3.],
         [3., 2., 1., 2.],
         [4., 3., 2., 1.]]
    """
    matrix = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(num_classes):
            distance = abs(i - j)
            matrix[i][j] = 1 + alpha * (distance ** k)    
    matrix = tf.cast(matrix, tf.float32)
    return matrix

def generate_exponential_penalty(num_classes=4, beta=2.0):
    """
    Generates a penalty matrix where penalties increase exponentially with class distance.
    W[i,j] = β^|i-j| 

    Args:
        num_classes (int): Number of classes.
        beta (float): Base of the exponential function. Default: 2.0.

    Returns:
        np.ndarray: Penalty matrix of shape (num_classes, num_classes).

    Example:
        >>> generate_exponential_penalty(4, beta=2.0)
        [[1., 2., 4., 8.],
         [2., 1., 2., 4.],
         [4., 2., 1., 2.],
         [8., 4., 2., 1.]]
    """
    matrix = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(num_classes):
            distance = abs(i - j)
            # Exponential penalty: beta^distance
            matrix[i][j] = beta ** distance
    matrix = tf.cast(matrix, tf.float32)
    return matrix

weight_matrix = None
def custom_loss(y_true, y_pred):
    
    # Standardni kros-entropija
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Konvertujemo sve u int32 da budu istog tipa
    y_true = tf.cast(y_true, tf.int32)
    y_pred_labels = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)  # Eksplicitno castujemo u int32
    
    weights = tf.gather_nd(weight_matrix, tf.stack([y_true, y_pred_labels], axis=1))
    
    return loss * weights

def format_training_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        hrs = int(seconds) // 3600
        mins = (int(seconds) % 3600) // 60
        secs = int(seconds) % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d} (hh:mm:ss)"



# print(generate_power_penalty(4, alpha=1.0, k=2))
# print(generate_exponential_penalty(num_classes=4, beta=2.71))

# y_true = np.array([0, 1, 2, 0, 1, 2])  # true class labels
# y_scores = np.array([
#     [0.8, 0.1, 0.1],  # predicted probabilities
#     [0.2, 0.7, 0.1],
#     [0.7, 0.3, 0.6],
#     [0.7, 0.2, 0.1],
#     [0.1, 0.8, 0.1],
#     [0.1, 0.1, 0.8]])
# plot_multiclass_ovo_roc_pr_curves(y_true, y_scores, class_names=['A', 'B', 'C'])


# Primer upotrebe classification_report_with_specificity
# y_true = [0, 1, 1, 2, 2, 2]  # Prave vrednosti
# y_pred = [0, 0, 1, 2, 2, 1]  # Predviđene vrednosti

# print(classification_report_with_specificity(y_true, y_pred))




# import matplotlib.pyplot as plt

# ppg = [14835,14632,14401,14134,13810,13423,12998,12536,12075,11622,11179,10773,10395,10016,9628,9222,8834,8484,8170,7856,7524,7127,6665,6176,5677,5160,4643,4117,3591,3055,2538,2049,1597,1181,812,480,166,-156,-507,-877,-1255,-1634,-1994,-2317,-2594,-2806,-2972,-3111,-3221,-3323,-3415,-3508,-3600,-3701,-3803,-3914,-4015,-4117,-4200,-4265,-4302,-4320,-4302,-4200,-3988,-3609,-2981,-1984,-544,1283,3341,5474,7570,9573,11438,13137,14669,16091,17402,18639,19783,20836,21787,22627,23347,23929,24372,24686,24889,24990,25000,24963,24870,24759,24612,24446,24270,24076,23873,23652,23430,23190,22950,22701,22433,22138,21824,21473,21094,20688,20236,19728,19174,18583,17965,17337,16700,16054,15371,14660,13912,13127,12324,11502,10681,9878,9093,8327,7570,6822,6093,5363,4662,3969,3305,2649,2012,1384,775,184,-396,-969,-1532,-2086,-2621,-3148,-3655,-4145,-4625,-5086,-5557,-6037,-6545,-7062,-7542,-7967,-8290,-8521,-8668,-8742,-8659,-8327,-7653,-6619,-5262,-3646,-1827,166,2289,4514,6766,8945,10939,12712,14272,15620,16792,17799,18666,19433,20097,20624,21002,21233,21334,21353,21325,21251,21150,21002,20827,20633,20430,20236,20051,19913,19811,19737,19636,19460,19183,18814,18353,17817,17245,16700,16220,15814,15435,15057,14651,14244,13847,13478,13118,12767,12398,12019,11622,11226,10810,10385,9961,9527,9093,8641,8179,7699,7210,6711,6213,5705,5188,4671,4154,3628,3120,2612,2114,1615,1126,637,129,-396,-941,-1495,-2040,-2575,-3101,-3646,-4191,-4735,-5243,-5696,-6065,-6351,-6517,-6545,-6370,-5936,-5188,-4163,-2926,-1541,-92,1375,2769,4062,5243,6333,7330,8234,9056,9776,10404,10958,11429,11853,12195,12463,12620,12684,12675,12620,12527,12389,12204,11983,11706,11382,11004,10579,10081,9508,8871,8225,7653,7191,6831,6517,6213,5899,5585,5271,4975,4699,4431,4172,3923,3683,3452,3240,3046,2889,2769,2668,2584,2492,2409,2317,2234,2160,2077,1984,1855,1689,1504,1292,1070,830,581,313,36,-240,-526,-803,-1080,-1357,-1615,-1883,-2160,-2446,-2751,-3055,-3369,-3701,-4071,-4486,-4948,-5437,-5926,-6379,-6766,-7071,-7237,-7274,-7210,-7062,-6831,-6526,-6129,-5649,-5096,-4542,-4006,-3526,-3111,-2788,-2557,-2400,-2335,-2326,-2381,-2464,-2557,-2668,-2797,-2981,-3221,-3517,-3868,-4265,-4717,-5206,-5751,-6323,-6933,-7579,-8234,-8890,-9518,-10072,-10579,-11050,-11521,-11992,-12481,-12980,-13469,-13967,-14447,-14909,-15352,-15777,-16174,-16552,-16912,-17263,-17596,-17900,-18196,-18454,-18676,-18879,-19063,-19230,-19414,-19673,-20024,-20467,-20974,-21510,-22045,-22562,-23042,-23467,-23836,-24122,-24344,-24492,-24603,-24676,-24741,-24787,-24806,-24796,-24778,-24759,-24741,-24750,-24759,-24769,-24769,-24769,-24796,-24833,-24898,-24953,-24990,-25000,-24990,-24953,-24898,-24732,-24390,-23772,-22858,-21630,-20088,-18288,-16349,-14411,-12601,-10949,-9453,-8087,-6850,-5696,-4606,-3535,-2538,-1634,-849,-184,350,775,1098,1320,1449,1514,1514,1495,1477,1477,1486,1514,1560,1615,1661,1698,1726,1754,1781,1800,1818,1837,1855,1864,1846,1800,1735,1670,1597,1532,1458,1384,1283,1172,1061,960,858,747,637,507,360,212,55,-120,-295,-489,-701,-923,-1153,-1394,-1615,-1818,-2003,-2151,-2289,-2418,-2548,-2677,-2815,-2954,-3111,-3286,-3471,-3683,-3886,-4089,-4283,-4486,-4708,-4957,-5197,-5409,-5557,-5622,-5585,-5428,-5151,-4745,-4228,-3581,-2769,-1744,-470,978,2446,3785,4892,5853,6711,7551,8364,9148,9896,10607,11281,11899,12472,12952,13321,13552,13663,13690,13681,13644,13580,13506,13423,13321,13220,13118,13026,12933,12860,12795,12730,12656,12564,12453,12343,12250,12158,12056,11927,11779,11613,11438,11281,11124,10958,10764,10552,10321,10062,9795,9490,9167,8807,8428,8040,7653,7274,6933,6610,6314,6046,5779,5511,5234,4948,4652,4357,4062,3775,3517,3258,3000,2723,2427,2114,1809,1514,1209,886,544,193,-138,-452,-720,-923,-1052,-1080,-978,-720,-258,406,1274,2326,3545,4828,6102,7293,8428,9545,10699,11853,12943,13903,14706,15380,15961,16469,16912,17309,17651,17937,18149,18251,18260,18205,18140,18103,18094,18103,18094,18085,18048,17992,17909,17826,17752,17716,17706,17697,17669,17614,17549,17476,17420,17355,17235,17042,16792,16506,16220,15952,15685,15408,15112,14789,14429,14050,13653,13247,12841,12444,12047,11650,11253,10847,10441,10044,9656,9296,8964,8650,8354,8059,7745,7422,7071,6720,6370,6019,5677,5345,5022,4689,4329,3942,3545,3157,2778,2455,2206,2058,2031,2123,2354,2760,3378,4265,5419,6748,8133,9471,10745,11936,13053,14087,15057,15952,16783,17540,18233,18842,19359,19783,20116,20356,20494,20540,20513,20420,20310,20199,20107,20024,19931,19839,19737,19627,19525,19405,19276,19110,18906,18694,18482,18260,18048,17826,17586,17337,17069,16783,16469,16137,15768,15371,14937,14484,14014,13533,13081,12675]

# plt.plot(normalize_m1_1(nfFilt(15,100,5,sd5(nfFilt(40,100,5,ppg)))))
# plt.plot(normalize_m1_1(nfFilt(40,100,5,ppg)))

# plt.show()

#### CLASSIFICATION 
# import itertools
# from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
#                            average_precision_score, label_binarize)
# from itertools import cycle

# import numpy as np
# import matplotlib.pyplot as plt
# import itertools
# from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
#                            average_precision_score, label_binarize)

def convert_to_labels(y):
    """
    Convert one-hot encoded arrays to label arrays if needed.
    
    Parameters:
    -----------
    y : array-like
        Either one-hot encoded array or label array
        
    Returns:
    --------
    numpy.ndarray : Label array
    """
    y = np.array(y)
    
    # Check if input is one-hot encoded (2D array with binary values)
    if len(y.shape) == 2 and y.shape[1] > 1:
        # Check if it's actually one-hot (each row sums to 1 and contains only 0s and 1s)
        if np.allclose(y.sum(axis=1), 1) and np.all(np.isin(y, [0, 1])):
            return np.argmax(y, axis=1)
    
    # If it's already labels or not one-hot, return as is
    return y.flatten()

def convert_to_onehot(y, n_classes=None):
    """
    Convert label arrays to one-hot encoded arrays if needed.
    
    Parameters:
    -----------
    y : array-like
        Either one-hot encoded array or label array
    n_classes : int, optional
        Number of classes (inferred if not provided)
        
    Returns:
    --------
    numpy.ndarray : One-hot encoded array
    """
    y = np.array(y)
    
    # If already one-hot encoded, return as is
    if len(y.shape) == 2 and y.shape[1] > 1:
        if np.allclose(y.sum(axis=1), 1) and np.all(np.isin(y, [0, 1])):
            return y
    
    # Convert labels to one-hot
    y_labels = y.flatten()
    unique_classes = np.unique(y_labels)
    if n_classes is None:
        n_classes = len(unique_classes)
    
    return label_binarize(y_labels, classes=range(n_classes))

def generate_dummy_probabilities(y_true, y_pred, noise_level=0.1, random_seed=42):
    """
    Generate dummy probability scores for demonstration when actual probabilities aren't available.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    noise_level : float, default=0.1
        Amount of noise to add to make probabilities realistic
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    numpy.ndarray : Probability scores with shape (n_samples, n_classes)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    y_true_labels = convert_to_labels(y_true)
    y_pred_labels = convert_to_labels(y_pred)
    
    n_samples = len(y_true_labels)
    all_classes = np.unique(np.concatenate([y_true_labels, y_pred_labels]))
    n_classes = len(all_classes)
    
    # Create base probabilities
    y_scores = np.zeros((n_samples, n_classes))
    
    # Set high probability for predicted class
    for i, pred_class in enumerate(y_pred_labels):
        # Find the index of pred_class in all_classes
        class_idx = np.where(all_classes == pred_class)[0][0]
        
        # Set high probability for predicted class
        y_scores[i, class_idx] = 0.7 + np.random.normal(0, noise_level)
        
        # Add some probability to other classes
        for j in range(n_classes):
            if j != class_idx:
                y_scores[i, j] = np.random.uniform(0.05, 0.25)
    
    # Normalize to make proper probabilities
    y_scores = np.abs(y_scores)  # Ensure non-negative
    row_sums = y_scores.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    y_scores = y_scores / row_sums
    
    return y_scores

def plot_multiclass_roc_pr_curves_journal(y_true, y_scores=None, y_pred=None, class_names=None, 
                                        model_name='Model', figsize=(12, 5), save_fig=True, dpi=300):
    """
    Plot ROC and Precision-Recall curves for multiclass classification.
    Supports both one-hot encoded and label format inputs.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels (can be one-hot encoded or labels)
    y_scores : array-like, shape (n_samples, n_classes), optional
        Predicted probability scores. If None, will generate dummy probabilities from y_pred
    y_pred : array-like, optional
        Predicted class labels (used to generate dummy probabilities if y_scores is None)
    class_names : list, optional
        List of class names for legend
    model_name : str, default='Model'
        Name of the model for filename
    figsize : tuple, default=(12, 5)
        Figure size in inches
    save_fig : bool, default=True
        Whether to save the figure
    dpi : int, default=300
        Resolution for saved figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    # Convert inputs to appropriate formats
    y_true_labels = convert_to_labels(y_true)
    
    # Handle missing probability scores
    if y_scores is None:
        if y_pred is None:
            raise ValueError("Either y_scores or y_pred must be provided")
        print("Warning: No probability scores provided. Generating dummy probabilities for visualization.")
        y_pred_labels = convert_to_labels(y_pred)
        y_scores = generate_dummy_probabilities(y_true_labels, y_pred_labels)
    
    # Ensure y_scores is 2D and numpy array
    y_scores = np.array(y_scores)
    if len(y_scores.shape) == 1:
        raise ValueError("y_scores must be a 2D array with shape (n_samples, n_classes)")
    
    # Debug information
    print(f"Debug: y_scores shape: {y_scores.shape}")
    print(f"Debug: y_true_labels shape: {y_true_labels.shape}")
    print(f"Debug: unique classes: {np.unique(y_true_labels)}")
    
    # Binarize labels if needed
    unique_classes = np.unique(y_true_labels)
    n_classes = len(unique_classes)
    
    # Ensure y_scores has correct number of classes
    if y_scores.shape[1] != n_classes:
        print(f"Warning: Adjusting y_scores from {y_scores.shape[1]} to {n_classes} classes")
        if y_scores.shape[1] < n_classes:
            # Pad with small probabilities
            padding = np.full((y_scores.shape[0], n_classes - y_scores.shape[1]), 0.01)
            y_scores = np.hstack([y_scores, padding])
            # Renormalize
            y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)
        else:
            # Truncate to match number of classes
            y_scores = y_scores[:, :n_classes]
    
    y_true_bin = convert_to_onehot(y_true_labels, n_classes)
    
    # Ensure y_true_bin is 2D
    if len(y_true_bin.shape) == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)
    
    # Set class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_classes]
    
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute Precision-Recall curve and average precision for each class
    precision, recall, avg_precision = {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
    
    # Compute micro-average PR curve and average precision
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
    avg_precision["micro"] = average_precision_score(y_true_bin, y_scores, average="micro")
    
    # Set up the figure with scientific styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Define grayscale-friendly patterns and styles for B&W printing
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<']
    colors_bw = ['black', 'gray', 'darkgray', 'dimgray', 'lightgray', 'silver']
    
    # Plot ROC curves
    for i in range(n_classes):
        ax1.plot(fpr[i], tpr[i], 
                color=colors_bw[i % len(colors_bw)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                marker=markers[i % len(markers)],
                markersize=4,
                markevery=max(1, len(fpr[i])//10),
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Add micro-average
    ax1.plot(fpr["micro"], tpr["micro"],
            color='red', linestyle='-', linewidth=3,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
    
    # Reference line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random classifier')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot PR curves
    for i in range(n_classes):
        ax2.plot(recall[i], precision[i],
                color=colors_bw[i % len(colors_bw)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                marker=markers[i % len(markers)],
                markersize=4,
                markevery=max(1, len(recall[i])//10),
                label=f'{class_names[i]} (AP = {avg_precision[i]:.3f})')
    
    # Add micro-average
    ax2.plot(recall["micro"], precision["micro"],
            color='red', linestyle='-', linewidth=3,
            label=f'Micro-average (AP = {avg_precision["micro"]:.3f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        filename = f"{model_name.replace(' ', '_')}_ROC_PR_curves.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved as: {filename}")
    
    plt.show()
    return fig

def plot_multiclass_ovo_roc_pr_curves_journal(y_true, y_scores=None, y_pred=None, class_names=None, 
                                            model_name='Model', figsize=(14, 6), save_fig=True, dpi=300):
    """
    Plot ROC and Precision-Recall curves for One-vs-One multiclass classification.
    Supports both one-hot encoded and label format inputs.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels (can be one-hot encoded or labels)
    y_scores : array-like, shape (n_samples, n_classes), optional
        Predicted probability scores. If None, will generate dummy probabilities from y_pred
    y_pred : array-like, optional
        Predicted class labels (used to generate dummy probabilities if y_scores is None)
    class_names : list, optional
        List of class names
    model_name : str, default='Model'
        Name of the model for filename
    figsize : tuple, default=(14, 6)
        Figure size in inches
    save_fig : bool, default=True
        Whether to save the figure
    dpi : int, default=300
        Resolution for saved figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    # Convert inputs to appropriate formats
    y_true_labels = convert_to_labels(y_true)
    
    # Handle missing probability scores
    if y_scores is None:
        if y_pred is None:
            raise ValueError("Either y_scores or y_pred must be provided")
        print("Warning: No probability scores provided. Generating dummy probabilities for visualization.")
        y_pred_labels = convert_to_labels(y_pred)
        y_scores = generate_dummy_probabilities(y_true_labels, y_pred_labels)
    
    # Ensure y_scores is 2D and numpy array
    y_scores = np.array(y_scores)
    if len(y_scores.shape) == 1:
        raise ValueError("y_scores must be a 2D array with shape (n_samples, n_classes)")
    
    # Debug information
    print(f"Debug OvO: y_scores shape: {y_scores.shape}")
    print(f"Debug OvO: y_true_labels shape: {y_true_labels.shape}")
    print(f"Debug OvO: unique classes: {np.unique(y_true_labels)}")
    
    unique_classes = np.unique(y_true_labels)
    n_classes = len(unique_classes)
    
    # Ensure y_scores has correct number of classes
    if y_scores.shape[1] != n_classes:
        print(f"Warning: Adjusting y_scores from {y_scores.shape[1]} to {n_classes} classes")
        if y_scores.shape[1] < n_classes:
            # Pad with small probabilities
            padding = np.full((y_scores.shape[0], n_classes - y_scores.shape[1]), 0.01)
            y_scores = np.hstack([y_scores, padding])
            # Renormalize
            y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)
        else:
            # Truncate to match number of classes
            y_scores = y_scores[:, :n_classes]

    # Set class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_classes]

    # Define styles for B&W printing
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors_bw = ['black', 'gray', 'darkgray', 'dimgray', 'lightgray', 'silver', 'gainsboro', 'whitesmoke']
    
    # Create class pair combinations for OvO analysis
    class_pairs = list(itertools.combinations(range(n_classes), 2))

    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, avg_precision = {}, {}, {}

    # Calculate metrics for each class pair
    for (class1, class2) in class_pairs:
        if class1 not in unique_classes or class2 not in unique_classes:
            print(f"Warning: Skipping {class_names[class1]} vs {class_names[class2]} due to missing class")
            continue

        # Filter data for only two classes
        mask = (y_true_labels == class1) | (y_true_labels == class2)
        y_true_filtered = y_true_labels[mask]
        y_scores_filtered = y_scores[mask, class1]
        
        y_true_bin_filtered = np.where(y_true_filtered == class1, 1, 0)

        # Check if both classes are present
        unique_vals = np.unique(y_true_bin_filtered)
        if len(unique_vals) < 2:
            print(f"Warning: Skipping {class_names[class1]} vs {class_names[class2]} due to uniform labels")
            continue

        # Calculate ROC and PR metrics
        fpr[(class1, class2)], tpr[(class1, class2)], _ = roc_curve(y_true_bin_filtered, y_scores_filtered, pos_label=1)
        roc_auc[(class1, class2)] = auc(fpr[(class1, class2)], tpr[(class1, class2)])

        precision[(class1, class2)], recall[(class1, class2)], _ = precision_recall_curve(y_true_bin_filtered, y_scores_filtered)
        avg_precision[(class1, class2)] = average_precision_score(y_true_bin_filtered, y_scores_filtered)

    # Print results in CSV format for scientific reporting
    print("\nOne-vs-One Classification Results:")
    print("Class Pair,ROC AUC,Average Precision")
    print("-" * 50)
    for (class1, class2) in class_pairs:
        if (class1, class2) in roc_auc:
            print(f"{class_names[class1]} vs {class_names[class2]},{roc_auc[(class1, class2)]:.4f},{avg_precision[(class1, class2)]:.4f}")

    # Set up the figure with scientific styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot ROC curves
    for idx, (class1, class2) in enumerate(class_pairs):
        if (class1, class2) in roc_auc:
            ax1.plot(fpr[(class1, class2)], tpr[(class1, class2)], 
                    color=colors_bw[idx % len(colors_bw)],
                    linestyle=line_styles[idx % len(line_styles)],
                    linewidth=2,
                    marker=markers[idx % len(markers)],
                    markersize=4,
                    markevery=max(1, len(fpr[(class1, class2)])//10),
                    label=f"{class_names[class1]} vs {class_names[class2]} (AUC = {roc_auc[(class1, class2)]:.3f})")

    # Reference line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    ax1.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    ax1.set_title("One-vs-One ROC Curves", fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Plot Precision-Recall curves
    for idx, (class1, class2) in enumerate(class_pairs):
        if (class1, class2) in avg_precision:
            ax2.plot(recall[(class1, class2)], precision[(class1, class2)], 
                    color=colors_bw[idx % len(colors_bw)],
                    linestyle=line_styles[idx % len(line_styles)],
                    linewidth=2,
                    marker=markers[idx % len(markers)],
                    markersize=4,
                    markevery=max(1, len(recall[(class1, class2)])//10),
                    label=f"{class_names[class1]} vs {class_names[class2]} (AP = {avg_precision[(class1, class2)]:.3f})")

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel("Recall", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Precision", fontsize=12, fontweight='bold')
    ax2.set_title("One-vs-One Precision-Recall Curves", fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        filename = f"{model_name.replace(' ', '_')}_OvO_ROC_PR_curves.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved as: {filename}")

    plt.show()
    return fig


def plot_confusion_matrix_journal(y_true, y_pred, classes=None, model_name='Model', 
                         figsize=(8, 6), save_fig=True, dpi=300, 
                         normalize=False, show_percentages=True):
    """
    Plot confusion matrix optimized for scientific publication with B&W printing support.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    classes : list, optional
        List of class names for labels
    model_name : str, default='Model'
        Name of the model for filename
    figsize : tuple, default=(8, 6)
        Figure size in inches
    save_fig : bool, default=True
        Whether to save the figure
    dpi : int, default=300
        Resolution for saved figure
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    show_percentages : bool, default=True
        Whether to show percentages alongside counts
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    cm : numpy.ndarray
        The confusion matrix
    """
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Set class names if not provided
    if classes is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        classes = [f'Class {i}' for i in unique_labels]
    
    # Normalize if requested
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_norm
        fmt_string = '.3f'
        title_suffix = ' (Normalized)'
    else:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm
        fmt_string = 'd'
        title_suffix = ''
    
    # Set up the figure with scientific styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grayscale colormap for B&W printing compatibility
    # Use white-to-black gradient instead of blue
    cmap = plt.cm.Greys
    
    # Create the heatmap
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, aspect='equal')
    
    # Add colorbar with proper formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if normalize:
        cbar.set_label('Normalized Count', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    else:
        cbar.set_label('Count', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_display.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            if normalize:
                if show_percentages:
                    text = f'{cm[i, j]}\n({cm_norm[i, j]:.1%})'
                else:
                    text = f'{cm_norm[i, j]:.3f}'
            else:
                if show_percentages and cm.sum(axis=1)[i] > 0:
                    percentage = cm[i, j] / cm.sum(axis=1)[i] * 100
                    text = f'{cm[i, j]}\n({percentage:.1f}%)'
                else:
                    text = f'{cm[i, j]}'
            
            # Choose text color based on background intensity
            color = "white" if cm_display[i, j] > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", 
                   color=color, fontsize=10, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix{title_suffix}', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_ylim(-0.5, len(classes) - 0.5)
    
    # Add thin grid lines between cells
    for i in range(len(classes) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1)
        ax.axvline(i - 0.5, color='white', linewidth=1)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        suffix = '_normalized' if normalize else ''
        filename = f"{model_name.replace(' ', '_')}_confusion_matrix{suffix}.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Confusion matrix saved as: {filename}")
    
    plt.show()
    
    return fig, cm

def plot_detailed_confusion_matrix_journal(y_true, y_pred, classes=None, model_name='Model',
                                 figsize=(12, 5), save_fig=True, dpi=300):
    """
    Plot both normalized and unnormalized confusion matrices side by side.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    classes : list, optional
        List of class names for labels
    model_name : str, default='Model'
        Name of the model for filename
    figsize : tuple, default=(12, 5)
        Figure size in inches
    save_fig : bool, default=True
        Whether to save the figure
    dpi : int, default=300
        Resolution for saved figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Set class names if not provided
    if classes is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        classes = [f'Class {i}' for i in unique_labels]
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Set up the figure
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Grayscale colormap for B&W printing
    cmap = plt.cm.Greys
    
    # Plot unnormalized confusion matrix
    im1 = ax1.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Count', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    # Set ticks and labels for first subplot
    ax1.set_xticks(np.arange(len(classes)))
    ax1.set_yticks(np.arange(len(classes)))
    ax1.set_xticklabels(classes, fontsize=10)
    ax1.set_yticklabels(classes, fontsize=10)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations for unnormalized
    thresh1 = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            percentage = cm[i, j] / cm.sum(axis=1)[i] * 100 if cm.sum(axis=1)[i] > 0 else 0
            text = f'{cm[i, j]}\n({percentage:.1f}%)'
            color = "white" if cm[i, j] > thresh1 else "black"
            ax1.text(j, i, text, ha="center", va="center", 
                    color=color, fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Class', fontsize=11, fontweight='bold')
    ax1.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    
    # Plot normalized confusion matrix
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap=cmap, aspect='equal')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Normalized Count', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    # Set ticks and labels for second subplot
    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_xticklabels(classes, fontsize=10)
    ax2.set_yticklabels(classes, fontsize=10)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations for normalized
    thresh2 = cm_norm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = f'{cm_norm[i, j]:.3f}'
            color = "white" if cm_norm[i, j] > thresh2 else "black"
            ax2.text(j, i, text, ha="center", va="center", 
                    color=color, fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Class', fontsize=11, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
    
    # Add grid lines
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, len(classes) - 0.5)
        ax.set_ylim(-0.5, len(classes) - 0.5)
        for i in range(len(classes) + 1):
            ax.axhline(i - 0.5, color='white', linewidth=1)
            ax.axvline(i - 0.5, color='white', linewidth=1)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        filename = f"{model_name.replace(' ', '_')}_confusion_matrix_detailed.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Detailed confusion matrix saved as: {filename}")
    
    plt.show()
    
    return fig

def calculate_specificity_per_class(y_true, y_pred, classes=None):
    """
    Calculate specificity (true negative rate) for each class in multi-class classification.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like  
        Predicted class labels
    classes : list, optional
        List of class names
        
    Returns:
    --------
    specificity_per_class : numpy.ndarray
        Specificity for each class
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    specificity_per_class = np.zeros(n_classes)
    
    for i in range(n_classes):
        # True negatives: all correct predictions except for class i
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        # False positives: incorrectly predicted as class i  
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        # Specificity = TN / (TN + FP)
        if (tn + fp) > 0:
            specificity_per_class[i] = tn / (tn + fp)
        else:
            specificity_per_class[i] = 0.0
            
    return specificity_per_class

def calculate_specificity_macro_micro(y_true, y_pred):
    """
    Calculate macro and micro averaged specificity.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
        
    Returns:
    --------
    tuple: (specificity_macro, specificity_micro)
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    # Calculate per-class specificity
    specificity_per_class = calculate_specificity_per_class(y_true, y_pred)
    
    # Macro: average of per-class specificities
    specificity_macro = np.mean(specificity_per_class)
    
    # Micro: calculate overall TN and FP
    total_tn = 0
    total_fp = 0
    
    for i in range(n_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        total_tn += tn
        total_fp += fp
    
    if (total_tn + total_fp) > 0:
        specificity_micro = total_tn / (total_tn + total_fp)
    else:
        specificity_micro = 0.0
        
    return specificity_macro, specificity_micro

def generate_confusion_matrix_report_journal(y_true, y_pred, classes=None, model_name='Model',                                             
                                   save_figs=True, dpi=300):
    """
    Generate comprehensive confusion matrix analysis with metrics table including specificity.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    classes : list, optional
        List of class names
    model_name : str, default='Model'
        Name of the model for filenames
    save_figs : bool, default=True
        Whether to save figures
    dpi : int, default=300
        Resolution for saved figures
        
    Returns:
    --------
    dict : Dictionary containing metrics and figures
    """
    
    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX ANALYSIS: {model_name}")
    print(f"{'='*60}")
    
    # Set class names if not provided
    if classes is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        classes = [f'Class {i}' for i in unique_labels]
    
    # Generate confusion matrix plots
    fig1, cm = plot_confusion_matrix_journal(y_true, y_pred, classes, model_name, 
                                   save_fig=save_figs, dpi=dpi, normalize=False)
    
    fig2 = plot_detailed_confusion_matrix_journal(y_true, y_pred, classes, model_name,
                                        save_fig=save_figs, dpi=dpi)
    
    # Calculate detailed metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Calculate specificity metrics
    specificity_macro, specificity_micro = calculate_specificity_macro_micro(y_true, y_pred)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    specificity_per_class = calculate_specificity_per_class(y_true, y_pred, classes)
    
    # Print summary table
    print(f"\nOverall Performance Metrics:")
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<20} {accuracy:.4f}")
    print(f"{'Precision (Macro)':<20} {precision_macro:.4f}")
    print(f"{'Recall (Macro)':<20} {recall_macro:.4f}")
    print(f"{'F1-Score (Macro)':<20} {f1_macro:.4f}")
    print(f"{'Specificity (Macro)':<20} {specificity_macro:.4f}")
    print(f"{'Precision (Micro)':<20} {precision_micro:.4f}")
    print(f"{'Recall (Micro)':<20} {recall_micro:.4f}")
    print(f"{'F1-Score (Micro)':<20} {f1_micro:.4f}")
    print(f"{'Specificity (Micro)':<20} {specificity_micro:.4f}")
    
    # Per-class metrics table
    print(f"\nPer-Class Performance Metrics:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Specificity':<12} {'Support':<10}")
    print("-" * 75)
    
    for i, class_name in enumerate(classes):
        support = np.sum(y_true == i)
        print(f"{class_name:<15} {precision_per_class[i]:.4f}     {recall_per_class[i]:.4f}     {f1_per_class[i]:.4f}     {specificity_per_class[i]:.4f}       {support:<10}")
    
    # Confusion matrix statistics
    print(f"\nConfusion Matrix Statistics:")
    print(f"Total samples: {len(y_true)}")
    print(f"Correctly classified: {np.trace(cm)} ({np.trace(cm)/len(y_true)*100:.2f}%)")
    print(f"Misclassified: {len(y_true) - np.trace(cm)} ({(len(y_true) - np.trace(cm))/len(y_true)*100:.2f}%)")
    
    # Additional metric explanations
    print(f"\nMetric Definitions:")
    print(f"- Precision: TP / (TP + FP) - Of all positive predictions, how many were correct?")
    print(f"- Recall (Sensitivity): TP / (TP + FN) - Of all actual positives, how many were found?")
    print(f"- Specificity: TN / (TN + FP) - Of all actual negatives, how many were correctly identified?")
    print(f"- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)")
    print(f"- Macro Average: Unweighted mean of per-class metrics")
    print(f"- Micro Average: Calculated globally by counting total TP, FP, FN")
    
    return {
        'confusion_matrix': cm,
        'figures': [fig1, fig2],
        'metrics': {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'specificity_macro': specificity_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'specificity_micro': specificity_micro,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'specificity_per_class': specificity_per_class
        }
    }


# y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
# y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2, 0])
# classes = ['Class A', 'Class B', 'Class C']

# results = generate_confusion_matrix_report_journal(
#     y_true, y_pred, 
#     classes=classes, 
#     model_name='Random Forest',
#     save_figs=True
# )

# print(f"\nGenerated confusion matrix report:{results}" )


# # Plot with dummy probabilities (when you only have predictions)
# fig1 = plot_multiclass_roc_pr_curves_journal(
#     y_true=y_true, 
#     y_pred=y_pred,
#     class_names=classes,
#     model_name='Example Model',
#     save_fig=True
# )