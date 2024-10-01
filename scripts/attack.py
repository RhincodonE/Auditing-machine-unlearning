import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import sys

def load_and_fit_gaussian(file_path):
    """Load observations from a file and fit to a Gaussian distribution."""
    data = pd.read_csv(file_path)['0']
    mean, std = norm.fit(data)
    max = np.min(data)
    return max, std

def determine_model(fitted_params, in_params, out_params, fpr_threshold=0.01):
    """
    Determine the model ('In' or 'Out') based on the ratio of PDFs and a threshold.

    :param fitted_params: Mean and standard deviation of the fitted distribution.
    :param in_params: Mean and standard deviation of the 'In' distribution.
    :param out_params: Mean and standard deviation of the 'Out' distribution.
    :param fpr_threshold: The desired false positive rate for thresholding.
    :return: 'In' or 'Out' depending on the ratio of the PDFs.
    """
    in_mean, in_std = in_params
    out_mean, out_std = out_params
    mean, std = fitted_params

    # Calculate the PDFs of the fitted mean for both 'In' and 'Out' distributions
    pdf_in = norm.pdf(mean, loc=in_mean, scale=in_std)
    pdf_out = norm.pdf(mean, loc=out_mean, scale=out_std)

    # Calculate the ratio of the PDFs
    pdf_ratio = pdf_in / pdf_out

    # Determine the threshold based on FPR = 0.01

    # Return 'In' if the ratio is above the threshold, otherwise 'Out'
    return pdf_ratio

def process_files(index, results_dir):
    """Process all files for a given index and store results."""
    in_params = (gaussian_params.loc['In', 'Mean'], gaussian_params.loc['In', 'Standard Deviation'])
    out_params = (gaussian_params.loc['Out', 'Mean'], gaussian_params.loc['Out', 'Standard Deviation'])
    results = []

    for label in ['in', 'out']:
        files = [f for f in os.listdir(results_dir) if f.startswith(f'{label}_observations_{index}_')]
        for file in files:
            file_path = os.path.join(results_dir, file)
            fitted_params = load_and_fit_gaussian(file_path)
            predicted_label = determine_model(fitted_params, in_params, out_params)
            results.append({'True': label.capitalize(), 'Prediction': predicted_label})

    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <aug_index>")
        sys.exit(1)

    aug_index = sys.argv[1]
    gaussian_params_path = f'/users/home/parentfolder/Augment/result/gaussian_params_{aug_index}.csv'
    results_dir = './result/'

    # Load Gaussian parameters
    gaussian_params = pd.read_csv(gaussian_params_path).set_index('Model')

    # Process files and get results
    results = process_files(aug_index, results_dir)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'/users/home/parentfolder/Augment/attack_result/AUC_epsilon/label_predictions_{aug_index}.csv', index=False)

    print("Processing complete. Results saved.")
