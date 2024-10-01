import pandas as pd
import os

# Path to the directory containing your CSV files
results_dir_kl = "/users/home/parentfolder/Augment/attack_result/kl_divergence"

# Path to the score.csv file
score_file =  "/users/home/parentfolder/Augment/scores.csv"

# Load the score data
scores_df = pd.read_csv(score_file)

# List to hold the results
results_kl = []

# Loop through each file in the results directory
for file in os.listdir(results_dir_kl):
    if file.startswith("label_predictions_KL_") and file.endswith(".csv"):
        # Extract the index from the file name
        index = int(file.split('_')[-1].replace('.csv', ''))

        # Read the current KL file
        kl_df = pd.read_csv(os.path.join(results_dir_kl, file))

        # Get the maximum KL value
        max_kl = kl_df['KL(in||out)'].max()

        # Get the corresponding score
        score = scores_df.loc[scores_df['Index'] == index, 'CorrectRate'].values[0]

        # Append the result
        results_kl.append({'index': index, 'max_kl': max_kl, 'score': score})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
results_df.to_csv(os.path.join(results_dir, "/users/home/parentfolder/Augment/attack_result/kl_and_scores_summary.csv"), index=False)

print("Completed processing all files. Results saved to kl_and_scores_summary.csv.")
