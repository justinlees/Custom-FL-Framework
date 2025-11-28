import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from fl_client import Net  # Import your model definition
import json
import os

# Configuration for the Attack
IMG_SIZE = (3, 28, 28)
BATCH_SIZE = 1  # Attacking a single image is easiest to visualize
ITERATIONS = 300 # How many optimization steps the attacker takes
ATTACK_LR = 0.1

def get_gradient(model, input_data, target_label):
    """Compute the gradient for a single training step."""
    output = model(input_data)
    loss = nn.CrossEntropyLoss()(output, target_label)
    return torch.autograd.grad(loss, model.parameters())

def reconstruct_image(model, target_gradient, device):
    """
    The Attack: Optimize a dummy image to match the target gradient.
    """
    # Start with a random noise image (what the attacker sees initially)
    dummy_data = torch.randn(BATCH_SIZE, *IMG_SIZE, device=device, requires_grad=True)
    
    # In many attacks, the label can be inferred or is known. We assume known label for worst-case.
    # (For PathMNIST, you'd get the label from the dataset)
    dummy_label = torch.tensor([0], device=device) 

    optimizer = torch.optim.LBFGS([dummy_data])
    
    print(f"  -> Running reconstruction attack ({ITERATIONS} iterations)...")
    
    for i in range(ITERATIONS):
        def closure():
            optimizer.zero_grad()
            output = model(dummy_data)
            loss = nn.CrossEntropyLoss()(output, dummy_label)
            dummy_gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            
            # Calculate distance between dummy gradient and target gradient
            grad_diff = 0
            for dg, tg in zip(dummy_gradient, target_gradient):
                grad_diff += ((dg - tg) ** 2).sum()
            
            grad_diff.backward()
            return grad_diff
        
        optimizer.step(closure)

    return dummy_data.detach()

def apply_framework_privacy_mechanism(gradient_tuple, config):
    """
    Simulate your framework's privacy (Clipping + Noise) manually for the audit.
    Handles both manual config keys and Opacus keys.
    """
    privacy_params = config.get('privacy_params', {})
    
    # 1. Determine Clipping Bound
    # Check for 'clipping_bound' (manual) first, then 'max_grad_norm' (Opacus)
    if 'clipping_bound' in privacy_params:
        bound = privacy_params['clipping_bound']
    elif 'max_grad_norm' in privacy_params:
        bound = privacy_params['max_grad_norm']
    else:
        bound = 4.0 # Default fallback
        print("Warning: No clipping bound found in config, using default 4.0")

    # 2. Determine Noise Multiplier
    # Check for 'noise_multiplier' (manual). If using Opacus with target_epsilon,
    # we don't have a static multiplier in config. We'll default to a reasonable value
    # for the audit simulation (e.g., 1.0) or try to estimate it.
    if 'noise_multiplier' in privacy_params:
        multiplier = privacy_params['noise_multiplier']
    else:
        # If we only have target_epsilon, we can't easily know the exact multiplier Opacus uses
        # without running the Accountant. For this visual audit, we'll assume a 
        # moderate multiplier to demonstrate the *effect* of noise.
        multiplier = 1.0 
        print(f"Notice: Using default noise_multiplier={multiplier} for audit (Opacus calculates this dynamically).")
    
    # --- Apply Mechanism ---
    
    # 1. Flatten all gradients to calculate global norm
    all_grads = torch.cat([g.flatten() for g in gradient_tuple])
    norm = torch.norm(all_grads)
    
    # 2. Clip (Global Norm Clipping)
    # scale = max_norm / norm, clamped at 1.0
    scale = min(1, bound / (norm + 1e-6))
    
    # 3. Add Noise
    noise_std = bound * multiplier
    
    noisy_grads = []
    for g in gradient_tuple:
        g_clipped = g * scale
        noise = torch.randn_like(g) * noise_std
        noisy_grads.append(g_clipped + noise)
        
    return tuple(noisy_grads)

def mse(img1, img2):
    return ((img1 - img2)**2).mean().item()

def run_audit():
    device = torch.device("cpu") # Keep it simple on CPU for this demo
    
    # 1. Load Data (Single Image)
    if not os.path.exists('pathmnist.npz'):
        print("Error: pathmnist.npz not found.")
        return
    
    try:
        data = np.load('pathmnist.npz')
        # Take the first image
        raw_img = data['train_images'][0] 
        real_label_int = data['train_labels'][0][0]
    except Exception as e:
         print(f"Error loading data: {e}")
         return
    
    # Transform to tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    real_data = transform(raw_img).unsqueeze(0).to(device)
    real_label = torch.tensor([real_label_int], device=device)

    # 2. Setup Model
    try:
        with open('config.json', 'r') as f: config = json.load(f)
    except FileNotFoundError:
        print("config.json not found, using defaults")
        config = {
             'model_architecture': {'num_classes': 9},
             'privacy_params': {'clipping_bound': 4.0, 'noise_multiplier': 0.4}
        }

    model = Net(num_classes=config['model_architecture']['num_classes']).to(device)
    model.eval() # We just need gradients

    # --- Scenario A: Attack on Non-Private Update ---
    print("\n--- Scenario A: Attack on Non-Private Update ---")
    raw_gradient = get_gradient(model, real_data, real_label)
    rec_img_A = reconstruct_image(model, raw_gradient, device)
    score_A = mse(real_data, rec_img_A)
    print(f"  Reconstruction MSE (Lower is better for attacker): {score_A:.4f}")
    
    # --- Scenario B: With Your Framework's Privacy ---
    print("\n--- Scenario B: Attack on Private Update ---")
    # Apply your specific privacy parameters using the local helper function
    private_gradient = apply_framework_privacy_mechanism(raw_gradient, config)
    rec_img_B = reconstruct_image(model, private_gradient, device)
    score_B = mse(real_data, rec_img_B)
    print(f"  Reconstruction MSE (Higher is better for privacy): {score_B:.4f}")

    # --- 4. Visualization ---
    # De-normalize for display (approximate)
    def show_img(tensor):
        img = tensor[0].permute(1, 2, 0).detach().numpy()
        img = (img * 0.5) + 0.5
        return np.clip(img, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(show_img(real_data))
    axs[0].set_title("Original Patient Image")
    axs[0].axis('off')
    
    axs[1].imshow(show_img(rec_img_A))
    axs[1].set_title(f"Recovered (No Privacy)\nMSE: {score_A:.4f}")
    axs[1].axis('off')
    
    axs[2].imshow(show_img(rec_img_B))
    axs[2].set_title(f"Recovered (With Privacy)\nMSE: {score_B:.4f}")
    axs[2].axis('off')
    
    # Helper for title
    pp = config.get('privacy_params', {})
    nm = pp.get('noise_multiplier', '1.0 (Audit Default)')
    cb = pp.get('clipping_bound', pp.get('max_grad_norm', 'N/A'))
    
    plt.suptitle(f"Privacy Audit: Gradient Inversion Attack\nNoise Mult: {nm}, Clip: {cb}")
    plt.tight_layout()
    plt.show()
    print("\nAudit complete. Displaying results...")

if __name__ == "__main__":
    run_audit()