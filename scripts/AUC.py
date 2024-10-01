import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, confusion_matrix

# Specify the directories containing the result files
result_dir_AUC = '/users/home/parentfolder/Augment/attack_result/AUC_epsilon'
results_dir_kl = "/users/home/parentfolder/Augment/attack_result/kl_divergence"
score_file = "/users/home/parentfolder/Augment/scores.csv"

# Load the score data
scores_df = pd.read_csv(score_file)

# Prepare to store AUC results
results_AUC = []

for filename in os.listdir(result_dir_AUC):
    if filename.endswith('.csv'):
        # Extract the file index from the filename
        file_index = filename.split('_')[-1].split('.')[0]

        score = scores_df.loc[scores_df['Index'] == int(file_index), 'CorrectRate'].values[0]

        # Load data from CSV
        data = pd.read_csv(os.path.join(result_dir_AUC, filename))

        # Extract true labels and prediction ratios (pdf_in/pdf_out)
        true_labels = data['True'].values
        predicted_ratios = data['Prediction'].values  # These are the ratios of pdf_in/pdf_out

        # Convert true labels to binary: 1 for 'In', 0 for 'Out'
        true_binary = np.array([1 if label == 'In' else 0 for label in true_labels])

        # Sort predicted ratios and find threshold at FPR = 0.001
        sorted_indices = np.argsort(predicted_ratios)[::-1]  # Sort descending
        sorted_ratios = predicted_ratios[sorted_indices]
        sorted_true_binary = true_binary[sorted_indices]

        # Calculate cumulative false positives and true negatives to determine FPR
        cumulative_fp = np.cumsum(1 - sorted_true_binary)  # Out samples
        cumulative_tn = np.sum(1 - true_binary)

        # Calculate FPR and find the threshold at FPR = 0.001
        fpr_values = cumulative_fp / cumulative_tn
        threshold_index = np.argmax(fpr_values >= 0.1)
        threshold = sorted_ratios[threshold_index]
        print(threshold)

        # Apply threshold to generate binary predictions: 1 if pdf_in/pdf_out > threshold, else 0
        predicted_binary = (predicted_ratios > threshold).astype(int)

        # Calculate AUC score
        auc_score = roc_auc_score(true_binary, predicted_binary)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_binary, predicted_binary).ravel()

        # Calculate FPR and TPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate ln(TPR/FPR) if both FPR and TPR are non-zero
        if fpr > 0 and tpr > 0:
            ln_tpr_fpr = np.log(tpr / fpr)
        else:
            ln_tpr_fpr = 'undefined'  # Handle division by zero or log(0) cases

        # Store the results
        results_AUC.append({
            'index': int(file_index),  # Ensure index is stored as an integer
            'auc': auc_score,
            'ln_tpr_fpr': ln_tpr_fpr,
            'TPR': tpr,
            'FPR': fpr,
            'Threshold': threshold,
        })

# Store KL divergence results
results_kl = []

# Loop through each file in the KL result directory
for file in os.listdir(results_dir_kl):
    if file.startswith("label_predictions_KL_") and file.endswith(".csv"):
        # Extract the index from the file name
        index = int(file.split('_')[-1].replace('.csv', ''))

        # Read the current KL file
        kl_df = pd.read_csv(os.path.join(results_dir_kl, file))

        # Get the maximum and average KL value
        max_kl = kl_df['KL(in||out)'].max()
        avg_kl = kl_df['KL(in||out)'].mean()

        # Get the corresponding score
        score = scores_df.loc[scores_df['Index'] == index, 'CorrectRate'].values[0]

        # Store the KL results
        results_kl.append({
            'index': int(index),  # Ensure index is stored as an integer
            'max_kl': max_kl,
            'avg_kl': avg_kl,
            'score': score
        })

# Convert both result lists into DataFrames
df_auc = pd.DataFrame(results_AUC)
df_kl = pd.DataFrame(results_kl)

# Ensure both 'index' columns are the same type (int64)
df_auc['index'] = df_auc['index'].astype(int)
df_kl['index'] = df_kl['index'].astype(int)

# Concatenate the two DataFrames on the 'index' column
final_df = pd.merge(df_auc, df_kl, on='index', how='outer')

# Save the concatenated DataFrame to a CSV file
final_df.to_csv('/users/home/parentfolder/Augment/attack_result/final_result.csv', index=False)

print("Results saved to final_result.csv.")
