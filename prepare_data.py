import os
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split
from geneformer import TranscriptomeTokenizer
import scipy.sparse as sp
import pickle
from utils import (
    plot_confusion,
    plot_cell_type_distribution,
    get_high_fraction_celltype_indices,
    download_model_and_dictionaries,
    calculate_and_plot_split_distributions,
    get_checkpoint_with_lowest_loss,
    compute_umap_random_subset
)

# Inputs 
REPO_ID = "ctheodoris/Geneformer"
MODEL_NAME = "gf-6L-30M-i2048"
PRETRAINED_DIR = "pretrained_models"  # Save in the current directory
DICT_BASE_URL = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/"
DICT_FILES = [
    "ensembl_mapping_dict_gc30M.pkl",
    "gene_median_dictionary_gc30M.pkl",
    "gene_name_id_dict_gc30M.pkl",
    "token_dictionary_gc30M.pkl"
]

# Download model and dictionaries
download_model_and_dictionaries(REPO_ID, MODEL_NAME, PRETRAINED_DIR, DICT_BASE_URL, DICT_FILES)

# Define base directories
BASE_DIR = os.path.abspath("outputs")  # All outputs will go into "outputs/"
DATA_DIR = os.path.abspath("data")  # Input data remains in the current directory
TOKENIZED_DATA_DIR = os.path.join(BASE_DIR, "tokenized_data")
RESULTS_DIR = os.path.join(BASE_DIR, "files")
MODEL_DIR = os.path.abspath(os.path.join(PRETRAINED_DIR, MODEL_NAME))
DICT_DIR = os.path.abspath(os.path.join(PRETRAINED_DIR, MODEL_NAME, "gene_dictionaries"))
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")  # Directory for saving embeddings
PLOT_DIR = os.path.join(BASE_DIR, "plots")  # New directory for plots

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DICT_DIR, exist_ok=True)
os.makedirs(TOKENIZED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Load data
cell_file = os.path.join(DATA_DIR, "cells.npy")
cells = np.load(cell_file, allow_pickle=True).ravel()[0]

expressions = cells["UMI"].toarray()
gene_names = cells["gene_ids"]
cell_types = cells["classes"]

# Optional chunk for data subsampling \ filtering
# Plot initial cell type distribution
plot_distr_1 = os.path.join(PLOT_DIR, "full_distribution.png")
plot_cell_type_distribution(cell_types, save_path=plot_distr_1)

# Filter by cell type fraction
high_fraction_indices = get_high_fraction_celltype_indices(cell_types, 0.05)
expressions = expressions[high_fraction_indices]
cell_types = cell_types[high_fraction_indices]

# Subsample data for efficient processing
_, subsample_indices = train_test_split(
    np.arange(len(cell_types)), test_size=0.01, stratify=cell_types, random_state=42
)
expressions = expressions[subsample_indices, :]
cell_types = cell_types[subsample_indices]

# Plot subsampled cell type distribution
plot_distr_2 = os.path.join(PLOT_DIR, "filtered_distribution.png")
plot_cell_type_distribution(cell_types, save_path=plot_distr_2)

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
adata_path = os.path.join(DATA_DIR, "adata.h5ad")
adata.write_h5ad(adata_path)
compute_umap_random_subset(adata, PLOT_DIR, subset_size=3000)

# Initialize Tokenizer
tokenizer = TranscriptomeTokenizer(
    custom_attr_name_dict={"cell_types": "cell_types", "cell_id": "cell_id"},
    model_input_size=2048,
    special_token=False,
    gene_median_file=os.path.join(DICT_DIR, "gene_median_dictionary_gc30M.pkl"),
    token_dictionary_file=os.path.join(DICT_DIR, "token_dictionary_gc30M.pkl"),
    gene_mapping_file=os.path.join(DICT_DIR, "ensembl_mapping_dict_gc30M.pkl")
)
# Run tokenizer
tokenizer.tokenize_data(
    data_directory=DATA_DIR,
    output_directory=TOKENIZED_DATA_DIR,
    output_prefix="my_dataset",
    file_format="h5ad",
    use_generator=False
)

# Train, validate, and test split
train_indices, temp_indices = train_test_split(
    np.arange(len(cell_types)), test_size=0.2, random_state=42, stratify=cell_types
)
cell_types_temp = cell_types[temp_indices]
eval_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, random_state=42, stratify=cell_types_temp
)

# Plot split distributions
calculate_and_plot_split_distributions(adata, train_indices, eval_indices, test_indices, PLOT_DIR)

# Prepare split dictionaries
train_ids = adata.obs["cell_id"].iloc[train_indices].tolist()
eval_ids = adata.obs["cell_id"].iloc[eval_indices].tolist()
test_ids = adata.obs["cell_id"].iloc[test_indices].tolist()
train_test_id_split_dict = {"attr_key": "cell_id", "train": train_ids + eval_ids, "test": test_ids}

split_dict_path = os.path.join(RESULTS_DIR, "train_test_id_split_dict.pkl")
with open(split_dict_path, "wb") as f:
    pickle.dump(train_test_id_split_dict, f)

train_eval_id_split_dict = {"attr_key": "cell_id", "train": train_ids, "eval": eval_ids}

split_dict_path = os.path.join(RESULTS_DIR, "train_eval_id_split_dict.pkl")
with open(split_dict_path, "wb") as f:
    pickle.dump(train_eval_id_split_dict, f)