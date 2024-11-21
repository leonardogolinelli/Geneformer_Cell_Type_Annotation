import os
import numpy as np
import scanpy as sc
from geneformer import TranscriptomeTokenizer
import scipy.sparse as sp
import pickle
from utils import *

# Inputs 
REPO_ID = "ctheodoris/Geneformer"
MODEL_NAME = "gf-6L-30M-i2048" # The smallest pretrained geneformer architecture
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
BASE_DIR = os.path.abspath("outputs") 
DATA_DIR = os.path.abspath("data") 
TOKENIZED_DATA_DIR = os.path.join(BASE_DIR, "tokenized_data")
RESULTS_DIR = os.path.join(BASE_DIR, "files")
MODEL_DIR = os.path.abspath(os.path.join(PRETRAINED_DIR, MODEL_NAME))
DICT_DIR = os.path.abspath(os.path.join(PRETRAINED_DIR, MODEL_NAME, "gene_dictionaries"))
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")
PLOT_DIR = os.path.join(BASE_DIR, "plots") 

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DICT_DIR, exist_ok=True)
os.makedirs(TOKENIZED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Load data
expressions, gene_names, cell_types = load_cell_data(DATA_DIR, file_name="cells.npy")

# Optional chunk for data subsampling \ filtering
expressions, cell_types = filter_and_subsample_data(
                    expressions, 
                    cell_types, 
                    fraction_threshold=0.05, 
                    subsample_fraction=0.01, 
                    plot_dir=PLOT_DIR
                    )

adata_path = os.path.join(DATA_DIR, "adata.h5ad")

# Create AnnData object
adata = prepare_adata(expressions, cell_types, gene_names, DATA_DIR, output_file="adata.h5ad")

# Compute UMAP of gene expression. Will be compared with embeddings UMAP later
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

#Split data intro training, evaluation, and test sets
# eval_test_size = (1-train_size) = (50%) eval (50%) train
train_indices, eval_indices, test_indices = split_data(cell_types, 
                                                       eval_test_size=0.2, 
                                                       random_state=42)


# Plot split distributions
calculate_and_plot_split_distributions(adata, train_indices, eval_indices, test_indices, PLOT_DIR)


# Build the dictionary for dataset preparation (train+eval vs test)..
# ..and the dictionary for fine tuning (train vs eval)
prepare_split_dicts(adata, train_indices, eval_indices, test_indices, RESULTS_DIR)