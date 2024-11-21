import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
import os
import urllib.request
from collections import Counter
import json
import glob
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


def download_model_and_dictionaries(repo_id, model_name, snapshot_dir, dict_base_url, dict_files):
    """
    Downloads the model snapshot and associated dictionary files.

    Parameters:
    - repo_id (str): The Hugging Face repository ID (e.g., "ctheodoris/Geneformer").
    - model_name (str): The name of the model (e.g., "gf-6L-30M-i2048").
    - snapshot_dir (str): The base directory where the snapshot will be saved.
    - dict_base_url (str): The base URL for downloading dictionary files.
    - dict_files (list of str): List of dictionary filenames to download.

    Returns:
    - None
    """
    # Ensure snapshot directory exists
    snapshot_dir = os.path.abspath(os.path.join(snapshot_dir))
    model_dir = os.path.abspath(os.path.join(snapshot_dir, model_name))
    if not os.path.exists(model_dir):
        print(f"Downloading snapshot for {repo_id}...")
        snapshot_download(repo_id, local_dir=snapshot_dir, allow_patterns=[f"{model_name}/*"])
        print(f"Snapshot saved in {snapshot_dir}")
    else:
        print(f"Snapshot already exists in {snapshot_dir}")
        return

    # Define dictionary directory
    dict_dir = os.path.join(snapshot_dir, model_name, "gene_dictionaries")

    # Ensure dictionary directory exists
    os.makedirs(dict_dir, exist_ok=True)

    # Download dictionary files
    for file in dict_files:
        output_file = os.path.join(dict_dir, file)
        if not os.path.exists(output_file):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(dict_base_url + file, output_file)
            print(f"Downloaded {file}")
        else:
            print(f"{file} already exists.")

    print("All files processed.")


def plot_confusion(conf_matrix, save_path=None):
    """
    Plots a normalized confusion matrix with consistent labels.

    Parameters:
    - conf_matrix: pd.DataFrame
        Confusion matrix with rows as true labels and columns as predicted labels.
    - save_path: str (optional)
        Path to save the confusion matrix plot.
    """
    # Normalize confusion matrix rows to sum to 1
    conf_matrix_normalized = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)

    # Add observation counts to labels
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)
    row_labels = [f"{label} ({int(count)})" for label, count in zip(conf_matrix.index, row_sums)]
    col_labels = [f"{label} ({int(count)})" for label, count in zip(conf_matrix.columns, col_sums)]

    print(f"row_labels: {row_labels}")
    print(f"col_labels: {col_labels}")
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                xticklabels=col_labels, yticklabels=row_labels)

    # Add axis labels and title
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    plt.title("Normalized Confusion Matrix with Observation Counts", fontsize=16)

    # Rotate x-axis labels by 45 degrees
    #plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()

    

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")

    # Show the plot
    plt.show()



def plot_cell_type_distribution(cell_types, save_path=None):
    """
    Plots the distribution of cell types as a bar chart with cell counts, 
    sorted by the number of cells, and optionally saves it to a file.
    
    Parameters:
        cell_types (array-like): Array of cell type labels.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
    """
    # Count occurrences of each class
    class_distribution = Counter(cell_types)

    # Sort by the number of cells in descending order
    sorted_distribution = dict(sorted(class_distribution.items(), key=lambda x: x[1], reverse=True))

    # Calculate relative frequencies
    total_count = sum(sorted_distribution.values())
    class_relative_frequencies = {key: value / total_count for key, value in sorted_distribution.items()}

    # Extract keys and values for plotting
    class_names = list(sorted_distribution.keys())
    class_frequencies = list(class_relative_frequencies.values())
    cell_counts = list(sorted_distribution.values())

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_frequencies, color="skyblue")

    # Add labels (number of cells) on top of the bars
    for bar, count in zip(bars, cell_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{count}", ha='center', va='bottom', fontsize=10)

    # Add labels and title
    plt.xlabel('Cell Types', fontsize=12)
    plt.ylabel('Relative Frequency', fontsize=12)
    plt.title('Cell Type Distribution (Relative Frequencies)', fontsize=14)

    # Rotate x-axis labels for better readability if needed
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Print relative frequencies for verification
    for key, value in sorted_distribution.items():
        print(f"{key}: {value}")

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved at: {save_path}")


def calculate_and_plot_split_distributions(adata, train_indices, eval_indices, test_indices, plot_dir):
    """
    Calculates and plots the cell type distribution for train, eval, and test splits.

    Parameters:
        adata (AnnData): AnnData object containing cell type information.
        train_indices (list): Indices of cells in the training split.
        eval_indices (list): Indices of cells in the evaluation split.
        test_indices (list): Indices of cells in the test split.
        plot_dir (str): Directory to save the plots.
    """
    # Extract cell types for each split
    train_cell_types = adata.obs["cell_types"].iloc[train_indices]
    eval_cell_types = adata.obs["cell_types"].iloc[eval_indices]
    test_cell_types = adata.obs["cell_types"].iloc[test_indices]

    # Plot overall and split distributions
    splits = {
        "Full_Data": adata.obs["cell_types"],
        "Train": train_cell_types,
        "Eval": eval_cell_types,
        "Test": test_cell_types,
    }

    for split_name, cell_types in splits.items():
        save_path = os.path.join(plot_dir, f"cell_type_distribution_{split_name.lower()}.png")
        plot_cell_type_distribution(cell_types, save_path=save_path)
        print(f"{split_name} cell type distribution plotted at: {save_path}")

def get_high_fraction_celltype_indices(cell_types, threshold):
    """
    Computes the fraction of cells for each cell type and returns the indices
    of cell types with a fraction lower than the given threshold.
    
    Parameters:
        cell_types (np.ndarray): A numpy array of cell types (categorical values).
        threshold (float): Fraction threshold to filter cell types.
        
    Returns:
        np.ndarray: Indices of cells belonging to cell types with a fraction lower than the threshold.
    """
    # Ensure input is a numpy array
    cell_types = np.array(cell_types)
    
    # Get unique cell types and their counts
    unique_cell_types, counts = np.unique(cell_types, return_counts=True)
    
    # Calculate fractions
    fractions = counts / len(cell_types)
    
    # Get cell types with fractions below the threshold
    low_fraction_cell_types = unique_cell_types[fractions >= threshold]
    
    # Find indices of cells with these cell types
    low_fraction_indices = np.where(np.isin(cell_types, low_fraction_cell_types))[0]
    
    return low_fraction_indices

def get_checkpoint_with_lowest_loss(validation_dir, current_date):
    # Locate all checkpoint directories
    checkpoint_dirs = glob.glob(
        os.path.join(validation_dir, f"{current_date}_geneformer_cellClassifier_fine_tuned_model", "ksplit1", "checkpoint-*")
    )
    
    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoint directories found. Ensure validation completed successfully.")
    
    # Dictionary to store loss values for each checkpoint
    checkpoint_losses = {}

    for checkpoint_dir in checkpoint_dirs:
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
                # Extract loss value (assuming it's stored under 'log_history' with key 'eval_loss')
                for entry in trainer_state.get("log_history", []):
                    if "eval_loss" in entry:
                        checkpoint_losses[checkpoint_dir] = entry["eval_loss"]
                        break
    
    if not checkpoint_losses:
        raise ValueError("No eval_loss found in trainer_state.json files. Check the validation process.")

    # Get the checkpoint with the minimum loss
    best_checkpoint = min(checkpoint_losses, key=checkpoint_losses.get)
    lowest_loss = checkpoint_losses[best_checkpoint]

    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Lowest eval loss: {lowest_loss}")
    return best_checkpoint



import scipy.sparse as sp

def compute_umap_random_subset(adata, plot_dir, subset_size=3000):
    """
    Computes UMAP on a random subset of cells and saves the plot.

    Parameters:
    - adata: AnnData
        The AnnData object containing the dataset.
    - plot_dir: str
        Directory where the UMAP plot will be saved.
    - subset_size: int, optional (default=3000)
        The number of random cells to select for UMAP computation.

    Returns:
    - None
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # Subset to 3000 randomly picked cells
    if adata.n_obs > subset_size:
        selected_indices = np.random.choice(adata.n_obs, subset_size, replace=False)
        adata_subset = adata[selected_indices].copy()
    else:
        adata_subset = adata.copy()
    
    # Preprocessing
    sc.pp.normalize_total(adata_subset, target_sum=1e4)
    sc.pp.log1p(adata_subset)
    sc.pp.highly_variable_genes(adata_subset, n_top_genes=2000, subset=True)
    sc.pp.scale(adata_subset, max_value=10)
    sc.pp.pca(adata_subset, n_comps=50)
    sc.pp.neighbors(adata_subset, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_subset)
    
    # Save UMAP plot
    umap_plot_path = os.path.join(plot_dir, "umap_random_subset_3000_cells.png")
    sc.pl.umap(
        adata_subset, 
        color="cell_types", 
        save="_random_subset_3000_cells.png", 
        show=False
    )
    
    # Move plot to specified plot directory
    umap_plot_file = "./figures/umap_random_subset_3000_cells.png"  # Scanpy default location
    if os.path.exists(umap_plot_file):
        os.rename(umap_plot_file, umap_plot_path)
        print(f"UMAP plot saved at: {umap_plot_path}")
    else:
        raise FileNotFoundError(f"UMAP plot was not generated as expected at: {umap_plot_file}")



def prepare_adata(expressions, cell_types, gene_names, data_dir, output_file="adata.h5ad"):
    """
    Prepares and saves an AnnData object.

    Parameters:
    - expressions : np.ndarray or scipy.sparse matrix
        Matrix of gene expression values.
    - cell_types : list or np.ndarray
        List of cell type labels for each observation (cell).
    - gene_names : list or np.ndarray
        List of gene names corresponding to columns in the expressions matrix.
    - data_dir : str
        Directory to save the AnnData file.
    - output_file : str (optional, default="adata.h5ad")
        Filename for the saved AnnData object.

    Returns:
    - adata : AnnData
        The prepared AnnData object.
    """
    # Create AnnData object
    adata = sc.AnnData(X=expressions)
    adata.obs["cell_types"] = cell_types
    adata.var_names = gene_names
    adata.var["ensembl_id"] = gene_names
    adata.obs["n_counts"] = adata.X.sum(1)
    adata.obs["cell_id"] = adata.obs_names.values

    # Sparsify data
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # Save AnnData
    os.makedirs(data_dir, exist_ok=True)
    adata_path = os.path.join(data_dir, output_file)
    adata.write_h5ad(adata_path)

    print(f"AnnData saved at: {adata_path}")
    return adata




def split_data(cell_types, eval_test_size=0.2, random_state=42):
    """
    Splits data indices into train, validation, and test sets.

    Parameters:
    - cell_types : np.ndarray or list
        Array of cell type labels for stratification.
    - test_size : float (default=0.2)
        Proportion of data to be held out for the test set.
    - eval_size : float (default=0.1)
        Proportion of the remaining data (after test split) to be held out for evaluation.
    - random_state : int (default=42)
        Random seed for reproducibility.

    Returns:
    - train_indices : np.ndarray
        Indices for the training set.
    - eval_indices : np.ndarray
        Indices for the evaluation set.
    - test_indices : np.ndarray
        Indices for the test set.
    """
    # First split: train and temporary (test + eval)
    train_indices, temp_indices = train_test_split(
        np.arange(len(cell_types)),
        test_size=eval_test_size,
        random_state=random_state,
        stratify=cell_types
    )

    # Get cell types for temporary split
    cell_types_temp = cell_types[temp_indices]

    # Second split: eval and test
    eval_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,  # Half of the temporary split goes to test
        random_state=random_state,
        stratify=cell_types_temp
    )

    return train_indices, eval_indices, test_indices


import os
import pickle

def prepare_split_dicts(adata, train_indices, eval_indices, test_indices, results_dir):
    """
    Prepares data split dictionaries for dataset preparation and fine-tuning steps.

    Parameters:
    - adata : AnnData
        Annotated data object containing cell IDs in `adata.obs["cell_id"]`.
    - train_indices : list or np.ndarray
        Indices for the training set.
    - eval_indices : list or np.ndarray
        Indices for the evaluation set.
    - test_indices : list or np.ndarray
        Indices for the test set.
    - results_dir : str
        Directory to save the split dictionaries.

    Returns:
    - train_test_id_split_dict : dict
        Dictionary for dataset preparation with train and test splits.
    - train_eval_id_split_dict : dict
        Dictionary for fine-tuning with train and eval splits.
    """
    # Prepare train, eval, and test IDs
    train_ids = adata.obs["cell_id"].iloc[train_indices].tolist()
    eval_ids = adata.obs["cell_id"].iloc[eval_indices].tolist()
    test_ids = adata.obs["cell_id"].iloc[test_indices].tolist()

    # Dataset preparation dictionary
    train_test_id_split_dict = {
        "attr_key": "cell_id",
        "train": train_ids + eval_ids,
        "test": test_ids,
    }
    train_test_split_path = os.path.join(results_dir, "train_test_id_split_dict.pkl")
    with open(train_test_split_path, "wb") as f:
        pickle.dump(train_test_id_split_dict, f)

    # Fine-tuning dictionary
    train_eval_id_split_dict = {
        "attr_key": "cell_id",
        "train": train_ids,
        "eval": eval_ids,
    }
    train_eval_split_path = os.path.join(results_dir, "train_eval_id_split_dict.pkl")
    with open(train_eval_split_path, "wb") as f:
        pickle.dump(train_eval_id_split_dict, f)

    print(f"Train-test split dictionary saved at: {train_test_split_path}")
    print(f"Train-eval split dictionary saved at: {train_eval_split_path}")

    return train_test_id_split_dict, train_eval_id_split_dict


def prepare_split_dicts(adata, train_indices, eval_indices, test_indices, results_dir):
    """
    Prepares data split dictionaries for dataset preparation and fine-tuning steps.

    Parameters:
    - adata : AnnData
        Annotated data object containing cell IDs in `adata.obs["cell_id"]`.
    - train_indices : list or np.ndarray
        Indices for the training set.
    - eval_indices : list or np.ndarray
        Indices for the evaluation set.
    - test_indices : list or np.ndarray
        Indices for the test set.
    - results_dir : str
        Directory to save the split dictionaries.

    Returns:
    - train_test_id_split_dict : dict
        Dictionary for dataset preparation with train and test splits.
    - train_eval_id_split_dict : dict
        Dictionary for fine-tuning with train and eval splits.
    """
    # Prepare train, eval, and test IDs
    train_ids = adata.obs["cell_id"].iloc[train_indices].tolist()
    eval_ids = adata.obs["cell_id"].iloc[eval_indices].tolist()
    test_ids = adata.obs["cell_id"].iloc[test_indices].tolist()

    # Dataset preparation dictionary
    train_test_id_split_dict = {
        "attr_key": "cell_id",
        "train": train_ids + eval_ids,
        "test": test_ids,
    }
    train_test_split_path = os.path.join(results_dir, "train_test_id_split_dict.pkl")
    with open(train_test_split_path, "wb") as f:
        pickle.dump(train_test_id_split_dict, f)

    # Fine-tuning dictionary
    train_eval_id_split_dict = {
        "attr_key": "cell_id",
        "train": train_ids,
        "eval": eval_ids,
    }
    train_eval_split_path = os.path.join(results_dir, "train_eval_id_split_dict.pkl")
    with open(train_eval_split_path, "wb") as f:
        pickle.dump(train_eval_id_split_dict, f)

    print(f"Train-test split dictionary saved at: {train_test_split_path}")
    print(f"Train-eval split dictionary saved at: {train_eval_split_path}")

    return train_test_id_split_dict, train_eval_id_split_dict


from sklearn.model_selection import train_test_split
import os

def filter_and_subsample_data(expressions, cell_types, fraction_threshold, subsample_fraction, plot_dir):
    """
    Filters and subsamples data based on cell type fraction and subsampling fraction.

    Parameters:
    - expressions : np.ndarray
        Expression data matrix (e.g., gene expression counts).
    - cell_types : np.ndarray or pd.Series
        Array or Series of cell type labels corresponding to each cell.
    - fraction_threshold : float
        Minimum fraction of cells for a cell type to be included in the dataset.
    - subsample_fraction : float
        Fraction of cells to keep after filtering.
    - plot_function : function
        Function to plot cell type distribution (e.g., `plot_cell_type_distribution`).
    - plot_dir : str
        Directory to save the distribution plots.

    Returns:
    - filtered_expressions : np.ndarray
        Filtered and subsampled expression data.
    - filtered_cell_types : np.ndarray
        Corresponding cell types after filtering and subsampling.
    """

    # Plot initial cell type distribution
    plot_before_filter = os.path.join(plot_dir, "before_filtering.png")
    plot_cell_type_distribution(cell_types, save_path=plot_before_filter)

    # Filter cell types by fraction threshold
    high_fraction_indices = get_high_fraction_celltype_indices(cell_types, fraction_threshold)
    filtered_expressions = expressions[high_fraction_indices]
    filtered_cell_types = cell_types[high_fraction_indices]

    # Subsample data for efficient processing
    _, subsample_indices = train_test_split(
        np.arange(len(filtered_cell_types)),
        test_size=subsample_fraction,
        stratify=filtered_cell_types,
        random_state=42
    )
    filtered_expressions = filtered_expressions[subsample_indices, :]
    filtered_cell_types = filtered_cell_types[subsample_indices]

    # Plot cell type distribution after filtering
    plot_after_filter = os.path.join(plot_dir, "after_filtering.png")
    plot_cell_type_distribution(filtered_cell_types, save_path=plot_after_filter)

    return filtered_expressions, filtered_cell_types



import os
import numpy as np

def load_cell_data(data_dir, file_name="cells.npy"):
    """
    Loads cell data from a .npy file and extracts expressions, gene names, and cell types.

    Parameters:
    - data_dir : str
        Directory containing the .npy file.
    - file_name : str, optional
        Name of the .npy file. Default is "cells.npy".

    Returns:
    - expressions : np.ndarray
        Dense matrix of expression data (e.g., UMI counts).
    - gene_names : np.ndarray
        Array of gene identifiers.
    - cell_types : np.ndarray
        Array of cell type labels corresponding to each cell.
    """
    # Construct full file path
    cell_file = os.path.join(data_dir, file_name)

    # Load cell data from .npy file
    cells = np.load(cell_file, allow_pickle=True).ravel()[0]

    # Extract components
    expressions = cells["UMI"].toarray()
    gene_names = cells["gene_ids"]
    cell_types = cells["classes"]

    return expressions, gene_names, cell_types
