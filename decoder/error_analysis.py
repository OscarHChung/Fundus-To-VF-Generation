import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions and true values (from evaluation)
# Suppose you have:
# all_preds, all_true = torch tensors of shape [num_samples, 72]
# mask_flat = boolean tensor of length 72 for points to ignore

def plot_pred_vs_true(all_preds, all_true, mask_flat=None, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Flatten tensors for plotting
    preds = all_preds.flatten()
    true = all_true.flatten()
    
    if mask_flat is not None:
        mask = mask_flat.repeat(all_preds.shape[0])  # repeat mask for all samples
        preds = preds[~mask]
        true = true[~mask]
    
    plt.figure(figsize=(6,6))
    plt.scatter(true, preds, alpha=0.5)
    plt.plot([0,1],[0,1], 'r--', label="Ideal")
    plt.xlabel("True VF (normalized)")
    plt.ylabel("Predicted VF (normalized)")
    plt.title("Predicted vs True VF values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pred_vs_true_scatter.png"))
    plt.close()
    print("Saved predicted vs true scatter plot.")

def plot_mae_heatmap(all_preds, all_true, mask_flat, grid_shape=(8,9), save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute MAE per point across all samples
    errors = torch.abs(all_preds - all_true)  # [num_samples, 72]
    errors_np = errors.mean(dim=0).cpu().numpy()  # [72]
    
    # Apply mask: optional, fill masked points with nan so they appear blank
    masked_errors = np.where(mask_flat.cpu().numpy(), np.nan, errors_np)
    
    heatmap_data = masked_errors.reshape(grid_shape)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="coolwarm", cbar_kws={'label': 'MAE'})
    plt.title("MAE Heatmap Across Visual Field")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mae_heatmap.png"))
    plt.close()
    print("Saved MAE heatmap.")

if __name__ == "__main__":
    # Example usage
    # Load from evaluation outputs
    eval_results = torch.load("evaluation_outputs.pt")  # contains dict: {'all_preds':..., 'all_true':...}
    all_preds = eval_results['all_preds']
    all_true = eval_results['all_true']
    
    # Use OD mask for example (adjust laterality accordingly)
    from decoder import mask_OD_flat
    mask = mask_OD_flat
    
    plot_pred_vs_true(all_preds, all_true, mask_flat=mask)
    plot_mae_heatmap(all_preds, all_true, mask_flat=mask)
