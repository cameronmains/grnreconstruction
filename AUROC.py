#AUROC

import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed(42)  # For reproducibility

# Load the edge list data
edge_list_data = pd.read_csv(r'C:\Users\crmai\Downloads\bipartite_GRN.csv')

def edge_list_to_adjacency_matrix(edge_list_df, n_tfs=100, n_targets=100):
    """
    Convert an edge list to an adjacency matrix.
    """
    # Create empty adjacency matrix
    adjacency_matrix = np.zeros((n_tfs, n_targets), dtype=bool)
    
    # Fill the adjacency matrix based on the edge list
    # Assuming edge_list_df has columns [0, 1] for TF ID and target ID respectively
    for _, row in edge_list_df.iterrows():
        tf_id = row[0]
        target_id = row[1]
        
        tf_idx = int(tf_id)
        target_idx = int(target_id) - 100  # Adjust for target ID offset
        
        # Check indices are within bounds
        if 0 <= tf_idx < n_tfs and 0 <= target_idx < n_targets:
            adjacency_matrix[tf_idx, target_idx] = True
        else:
            print(f"Warning: Edge ({tf_id}, {target_id}) is out of bounds")
    
    # Create row and column labels
    tf_labels = [f"TF_{i}" for i in range(n_tfs)]
    target_labels = [f"Gene_{i+100}" for i in range(n_targets)]
    
    # Convert to DataFrame with proper labels
    adjacency_df = pd.DataFrame(adjacency_matrix, 
                               index=tf_labels,
                               columns=target_labels)
    
    return adjacency_df

# Create the adjacency matrix
gold_standard = edge_list_to_adjacency_matrix(edge_list_data)

print("Gold standard shape:", gold_standard.shape)
print("Number of regulatory interactions:", gold_standard.values.sum())
print("Network density:", gold_standard.values.sum() / (gold_standard.shape[0] * gold_standard.shape[1]))

# Preview
print("\nFirst 5 TFs, first 5 targets:")
print(gold_standard.iloc[:5, :5])

# For AUROC calculation, you'd need a weights matrix with the same shape
def create_sample_weights_matrix(gold_standard, noise_level=0.3):
    """Create a sample weights matrix similar to GENIE3 output."""
    n_tfs, n_targets = gold_standard.shape
    
    # Initialize weights
    weights = np.random.rand(n_tfs, n_targets) * 0.3  # Low weights for non-edges
    
    # Assign higher weights to true edges (with some noise)
    true_edges = gold_standard.values
    weights[true_edges] = 0.7 + np.random.rand(np.sum(true_edges)) * 0.3
    
    # Create DataFrame with same index/columns as gold standard
    weights_df = pd.DataFrame(weights, 
                            index=gold_standard.index,
                            columns=gold_standard.columns)
    
    return weights_df

# Create sample weights matrix
weights_matrix = create_sample_weights_matrix(gold_standard)

# Now calculate AUROC
from sklearn.metrics import roc_auc_score

# Flatten matrices
y_true = gold_standard.values.flatten()
y_scores = weights_matrix.values.flatten()

# Calculate overall AUROC
overall_auroc = roc_auc_score(y_true, y_scores)
print(f"\nOverall AUROC: {overall_auroc:.4f}")

# Calculate AUROC per target gene
def calculate_target_auroc(weights_matrix, gold_standard):
    """Calculate AUROC for each target gene."""
    auroc_per_target = {}
    
    for col in gold_standard.columns:
        y_true = gold_standard[col].values
        y_scores = weights_matrix[col].values
        
        # Skip if all examples are from one class
        if np.all(y_true) or not np.any(y_true):
            auroc_per_target[col] = np.nan
            continue
            
        auroc = roc_auc_score(y_true, y_scores)
        auroc_per_target[col] = auroc
    
    return pd.Series(auroc_per_target)

target_auroc = calculate_target_auroc(weights_matrix, gold_standard)
print("\nAUROC per target gene (first 5):")
print(target_auroc.head())
print(f"Average AUROC across targets: {target_auroc.mean():.4f}")
