"""
This script constructs a curated dataset of photoplethysmogram (PPG) signals retrieved from a local web service.
It performs signal filtering, heartbeat segmentation, stability assessment, and demographic balancing to ensure
high-quality samples suitable for physiological modeling and machine learning applications.

Key features:
- Automated retrieval and validation of PPG data with demographic metadata
- Beat-level segmentation and normalization using detrending and min-max scaling
- Signal stability evaluation via intra- and inter-beat correlation metrics
- Controlled sampling per user and demographic group to ensure dataset diversity
- Periodic saving and summary statistics for reproducibility and monitoring

The final output is a CSV file containing raw signals, template beats, and metadata,
intended for use in research on cardiovascular dynamics, biometric modeling, and health analytics.
"""
import os
import requests
import numpy as np
import csv
import json
from scipy.signal import find_peaks, detrend
from scipy.stats import pearsonr
from util import nfFilt
from collections import defaultdict, Counter

# Constants
service_url = 'http://localhost/service/ppg.php?id='
start_id = 1300000
target_count = 10000
sampling_rate = 100
left_samples = 45
right_samples = 105
save_every = 100
output_folder = "data"
output_file = os.path.join(output_folder, "rws_ppg_regression_dataset.csv")

# Dataset and statistics
dataset = []
gender_count = {0: 0, 1: 0}
age_sex_distribution = defaultdict(int)
age_app_tracker = defaultdict(set)
user_sample_count = defaultdict(int)
user_age_signals = defaultdict(list)  # (app_id, age) → list of indices

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def get_beat_segments(signal, peaks):
    segments = []
    for peak in peaks:
        start = peak - left_samples
        end = peak + right_samples
        if start >= 0 and end < len(signal):
            seg = detrend(signal[start:end])
            seg = (seg - np.min(seg)) / (np.max(seg) - np.min(seg))
            segments.append(seg)
    return segments

def process_signal(signal, peaks):
    beats = get_beat_segments(signal, peaks)
    if not beats:
        return None
    avg = np.mean(beats, axis=0)
    beat_corrs = [pearsonr(seg, avg)[0] for seg in beats]
    weights = np.clip(beat_corrs, 0, 1)
    weighted_avg = np.average(beats, axis=0, weights=weights)
    beat_corrs = [pearsonr(seg, weighted_avg)[0] for seg in beats]
    corr_matrix = np.corrcoef(beats)
    beat_corr_ratio = np.sum(np.array(beat_corrs) > 0.8) / len(beat_corrs)
    interbeat_corr_ratio = np.sum(corr_matrix[~np.eye(len(corr_matrix), dtype=bool)] > 0.7) / (len(beats)**2 - len(beats))
    is_stable = beat_corr_ratio > 0.93 and interbeat_corr_ratio > 0.87
    return weighted_avg, is_stable, beat_corr_ratio, interbeat_corr_ratio

def save_dataset(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'age', 'sex', 'hr', 'rmssd', 'data_id', 'app_id',
            'ppg_signal', 'template_ppg', 'beat_corr_ratio', 'interbeat_corr_ratio'
        ])
        writer.writeheader()
        writer.writerows(data)

def print_stats(data):
    genders = Counter(int(row['sex']) for row in data)
    ages = Counter(int(row['age']) for row in data)
    print(f"\nDataset statistics ({len(data)} entries):")
    print(f"  Male: {genders.get(0, 0)}")
    print(f"  Female: {genders.get(1, 0)}")
    print(f"  Age distribution (top 5): {ages.most_common(5)}\n")

# Main loop
current_id = start_id
while len(dataset) < target_count:
    try:
        response = requests.get(f"{service_url}{current_id}")
        data = json.loads(response.text)[0]
        gender = int(data['gender'])  # 0 = male, 1 = female
        age = int(data['age'])
        ppg_raw = np.array(data['ppg'])

        if current_id % 100 == 0:
            print(f"Current ID: {current_id} | Count: {current_id - start_id} | Dataset size: {len(dataset)}")

        try:
            app_id = int(data.get('app_id', None))
            if app_id <= 1:
                current_id += 1
                continue
        except (ValueError, TypeError):
            current_id += 1
            continue

        if not (17 <= age <= 75) or len(ppg_raw) <= 1100:
            current_id += 1
            continue

        if user_sample_count[app_id] >= 10:
            current_id += 1
            continue

        age_sex_key = (age, gender)
        max_per_group = target_count // ((75 - 17 + 1) * 2)
        if age_sex_distribution[age_sex_key] >= max_per_group:
            current_id += 1
            continue

        ppg = nfFilt(13, 100, 5, ppg_raw)
        peaks, _ = find_peaks(ppg, distance=int(.7 * sampling_rate * 60 / int(data['hr'])))
        result = process_signal(ppg, peaks)

        if result is None:
            current_id += 1
            continue

        weighted_avg, is_stable, beat_corr_ratio, interbeat_corr_ratio = result
        if not is_stable:
            current_id += 1
            continue

        quality_score = 0.5 * beat_corr_ratio + 0.5 * interbeat_corr_ratio
        key = (app_id, age)
        existing_indices = user_age_signals[key]

        if len(existing_indices) >= 10:
            weakest_index = min(
                existing_indices,
                key=lambda i: 0.5 * float(dataset[i]['beat_corr_ratio']) +
                              0.5 * float(dataset[i]['interbeat_corr_ratio'])
            )
            weakest_score = 0.5 * float(dataset[weakest_index]['beat_corr_ratio']) + \
                            0.5 * float(dataset[weakest_index]['interbeat_corr_ratio'])

            if quality_score <= weakest_score:
                current_id += 1
                continue
            else:
                dataset.pop(weakest_index)
                user_age_signals[key].remove(weakest_index)
                age_sex_distribution[age_sex_key] -= 1
                gender_count[gender] -= 1
                user_sample_count[app_id] -= 1

        dataset.append({
            'id': data['id'],
            'age': age,
            'sex': gender,
            'hr': data['hr'],
            'rmssd': data['rmssd'],
            'data_id': data['data_id'],
            'app_id': app_id,
            'ppg_signal': ','.join(map(str, data['ppg'])),
            'template_ppg': ','.join(map(lambda x: f"{x:.5f}", weighted_avg)),
            'beat_corr_ratio': f"{beat_corr_ratio:.4f}",
            'interbeat_corr_ratio': f"{interbeat_corr_ratio:.4f}"
        })

        index = len(dataset) - 1
        user_age_signals[key].append(index)
        user_sample_count[app_id] += 1
        gender_count[gender] += 1
        age_sex_distribution[age_sex_key] += 1
        age_app_tracker[app_id].add(age)

        if len(dataset) % save_every == 0:
            save_dataset(output_file, dataset)
            print_stats(dataset)

    except Exception as e:
        print(f"Error at ID {current_id}: {e}")

    current_id += 1

# Final save
save_dataset(output_file, dataset)
print("Dataset successfully saved to:", output_file)
print_stats(dataset)