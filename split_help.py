import os
import torch

# Path to the folder containing the original .pt files with batched data
source_dir = "/home/yangqing/Documents/My_PointCloud_Model/data/modelnet_mixup"
# Path to the folder where you want to save the individual .pt files
destination_dir = "/home/yangqing/Documents/My_PointCloud_Model/data/modelnet_mixup_2"

# Create the destination folder if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Loop over each .pt file in the source directory
for file_name in os.listdir(source_dir):
    if file_name.endswith('.pt'):
        # Load the batched data
        file_path = os.path.join(source_dir, file_name)
        batched_data = torch.load(file_path)

        # Extract the point clouds and labels (assuming they are stored under 'pc' and 'label')
        point_clouds = batched_data['pc']  # Expected shape [32, 1024, 3]
        labels = batched_data['label']     # Expected shape [32]

        # Loop over each sample in the batch
        for i in range(point_clouds.shape[0]):
            # Prepare individual data for the current sample
            single_data = {
                'pc': point_clouds[i],         # Shape [1024, 3]
                'label': labels[i].item()      # Convert to scalar
            }

            # Save each sample to a new .pt file
            new_file_name = f"{os.path.splitext(file_name)[0]}_{i}.pt"
            new_file_path = os.path.join(destination_dir, new_file_name)
            torch.save(single_data, new_file_path)

            print(f"Saved {new_file_name} to {destination_dir}")
