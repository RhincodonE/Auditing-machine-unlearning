#!/bin/bash

# Base YAML configuration file path
yaml_file="./configs/config.yaml"
original_yaml_file="./configs/config_original.yaml"

# Load indices sorted by CorrectRate from a CSV
# Assuming the CSV is named 'correct_rate.csv' and is located in the current directory
mapfile -t aug_indices < <(awk -F, 'NR > 1 {print $1 "," $2}' scores.csv | sort -t ',' -k2,2rn | awk -F, '{print $1}' | awk 'NR%2==1')

# Make a copy of the original YAML to preserve it
cp $yaml_file $original_yaml_file

# Create necessary directories
mkdir -p /users/home/parentfolder/Augment/result
mkdir -p /users/home/parentfolder/Augment/tmp
mkdir -p /users/home/parentfolder/Augment/models
mkdir -p /users/home/parentfolder/Augment/attack_result
mkdir -p /users/home/parentfolder/Augment/attack_result/kl_divergence
mkdir -p /users/home/parentfolder/Augment/attack_result/AUC_epsilon

# Loop over each augmentation index
for aug_index in "${aug_indices[@]}"
do
    echo "Processing sample index: ${aug_index}"

    # Update the YAML file for the current index
    sed -i "s|sample_index:.*|sample_index: ${aug_index}|" $yaml_file
    sed -i "s|in_dataset: /users/home/parentfolder/Augment/tmp/.*|in_dataset: /users/home/parentfolder/Augment/tmp/cifar_in_${aug_index}.beton|" $yaml_file
    sed -i "s|out_dataset: /users/home/parentfolder/Augment/tmp/.*|out_dataset: /users/home/parentfolder/Augment/tmp/cifar_out_${aug_index}.beton|" $yaml_file
    sed -i "s|augment_dataset: /users/home/parentfolder/Augment/tmp/.*|augment_dataset: /users/home/parentfolder/Augment/tmp/cifar_augment_${aug_index}.beton|" $yaml_file
    sed -i "s|in_shadow_observations: /users/home/parentfolder/Augment/result/.*|in_shadow_observations: /users/home/parentfolder/Augment/result/in_shadow_observations_${aug_index}.csv|" $yaml_file
    sed -i "s|out_shadow_observations: /users/home/parentfolder/Augment/result/.*|out_shadow_observations: /users/home/parentfolder/Augment/result/out_shadow_observations_${aug_index}.csv|" $yaml_file
    sed -i "s|in_shadow_model_save_path: /users/home/parentfolder/Augment/models/.*|in_shadow_model_save_path: /users/home/parentfolder/Augment/models/model_in_shadow_${aug_index}.pth|" $yaml_file
    sed -i "s|out_shadow_model_save_path: /users/home/parentfolder/Augment/models/.*|out_shadow_model_save_path: /users/home/parentfolder/Augment/models/model_out_shadow_${aug_index}.pth|" $yaml_file

    # Preprocess and training with shadow models
    python /users/home/parentfolder/Augment/scripts/write_datasets.py --config-file $yaml_file
    python /users/home/parentfolder/Augment/scripts/train_shadow.py --config-file $yaml_file
    python /users/home/parentfolder/Augment/scripts/augment_loss_shadow.py --config-file $yaml_file
    python /users/home/parentfolder/Augment/scripts/get_distribution.py ${aug_index}

    # Inner loop to handle multiple iterations of model training and evaluation
    for i in {0..30}
    do
        echo "Iteration $i of model training and evaluation for index ${aug_index}"

        sed -i "s|in_observations: /users/home/parentfolder/Augment/result/.*|in_observations: /users/home/parentfolder/Augment/result/in_observations_${aug_index}_${i}.csv|" $yaml_file
        sed -i "s|out_observations: /users/home/parentfolder/Augment/result/.*|out_observations: /users/home/parentfolder/Augment/result/out_observations_${aug_index}_${i}.csv|" $yaml_file
        sed -i "s|in_model_save_path: /users/home/parentfolder/Augment/models/.*|in_model_save_path: /users/home/parentfolder/Augment/models/model_in_${aug_index}_${i}.pth|" $yaml_file
        sed -i "s|out_model_save_path: /users/home/parentfolder/Augment/models/.*|out_model_save_path: /users/home/parentfolder/Augment/models/model_out_${aug_index}_${i}.pth|" $yaml_file

        # Training and loss generation
        python /users/home/parentfolder/Augment/scripts/train_cifar.py --config-file $yaml_file
        python /users/home/parentfolder/Augment/scripts/augment_loss.py --config-file $yaml_file
        rm /users/home/parentfolder/Augment/models/model_in_${aug_index}_${i}.pth
        rm /users/home/parentfolder/Augment/models/model_out_${aug_index}_${i}.pth
    done

    # Perform attack and analyze
    python /users/home/parentfolder/Augment/scripts/attack.py ${aug_index}
    python /users/home/parentfolder/Augment/scripts/attack_distribution.py ${aug_index}
done

# Cleanup: Restore original configuration file and remove backup
cp $original_yaml_file $yaml_file
rm $original_yaml_file

echo "Privacy score generation finished."
