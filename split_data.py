import numpy as np
import os
import json

def split_dataset():
    # 1. Load configuration to get the number of clients
    config_path = 'config.json'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # We use 'clients_per_round' as the total number of clients to split data for
        # You could add a separate 'total_clients' field to config.json if you prefer
        num_clients = config.get('clients_per_round', 2) 
        print(f"Read 'clients_per_round' from config: {num_clients}")
    except Exception as e:
        print(f"Error reading config.json: {e}")
        return

    # 2. Load the master dataset
    if not os.path.exists('pathmnist.npz'):
        print("Error: pathmnist.npz not found. Please download it first.")
        return

    try:
        data = np.load('pathmnist.npz')
        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
    except Exception as e:
        print(f"Error loading pathmnist.npz: {e}")
        return
    
    total_train_samples = len(train_images)
    total_test_samples = len(test_images)
    
    # 3. Calculate split sizes
    train_split_size = total_train_samples // num_clients
    test_split_size = total_test_samples // num_clients
    
    print(f"Total Training Samples: {total_train_samples}")
    print(f"Splitting into {num_clients} clients (~{train_split_size} samples each).")

    # 4. Create the splits
    for i in range(num_clients):
        # Define indices
        train_start = i * train_split_size
        train_end = (i + 1) * train_split_size
        
        test_start = i * test_split_size
        test_end = (i + 1) * test_split_size
        
        # Slice the data
        client_train_images = train_images[train_start:train_end]
        client_train_labels = train_labels[train_start:train_end]
        
        client_test_images = test_images[test_start:test_end]
        client_test_labels = test_labels[test_start:test_end]
        
        # Define filename (e.g., hospital_1_data.npz)
        # Using 1-based indexing for friendly names
        filename = f"hospital_{i+1}_data.npz"
        
        # Save to new .npz file
        np.savez(filename, 
                 train_images=client_train_images, 
                 train_labels=client_train_labels,
                 test_images=client_test_images, 
                 test_labels=client_test_labels)
        
        print(f"Created {filename}: {len(client_train_images)} train samples")

if __name__ == "__main__":
    split_dataset()