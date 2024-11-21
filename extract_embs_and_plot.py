import os
import datetime
import pickle
import pandas as pd
from geneformer import EmbExtractor
from utils import get_checkpoint_with_lowest_loss  # Ensure these functions are correctly defined

# Define the base directory
BASE_DIR = os.path.abspath("outputs")  # Base directory for outputs
PLOT_DIR = os.path.join(BASE_DIR, "plots")  # Directory for plots
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")  # Directory for embeddings
TOKENIZED_DATA_DIR = os.path.join(BASE_DIR, "tokenized_data")  # Directory for tokenized data
RESULTS_DIR = os.path.join(BASE_DIR, "files")  # Directory for files
TEST_OUTPUT_DIR = os.path.join(RESULTS_DIR, "test_evaluation")  # Test evaluation directory

# Ensure directories exist
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Load `id_class_dict` from the previously prepared dictionary
id_class_dict_path = os.path.join(RESULTS_DIR, "cell_classifier_id_class_dict.pkl")
if not os.path.exists(id_class_dict_path):
    raise FileNotFoundError(f"Missing id_class_dict file: {id_class_dict_path}")
with open(id_class_dict_path, "rb") as f:
    id_class_dict = pickle.load(f)
print("Loaded id_class_dict:", id_class_dict)
print(id_class_dict)

# Obtain the best checkpoint path programmatically
validation_dir = os.path.join(RESULTS_DIR, "validation")
current_date = datetime.datetime.now().strftime("%y%m%d")
best_checkpoint_dir = get_checkpoint_with_lowest_loss(validation_dir, current_date)
best_checkpoint_path = os.path.abspath(best_checkpoint_dir)
print("Best checkpoint path:", best_checkpoint_path)

max_cells_to_plot = 3000

# Initialize EmbExtractor
cell_types_unique = list(id_class_dict.values())
embex = EmbExtractor(
    model_type="CellClassifier",
    num_classes=len(id_class_dict),
    emb_mode="cell",
    filter_data={"cell_types": cell_types_unique},
    max_ncells=max_cells_to_plot,
    emb_layer=-1,
    emb_label=["cell_types"],
    labels_to_plot=["cell_types"],
    forward_batch_size=10,  # Adjust for memory efficiency
    nproc=10,  # Number of processes
)

# Extract embeddings
embs_df, embs_tensor = embex.extract_embs(
    model_directory=best_checkpoint_path,
    input_data_file=os.path.join(TOKENIZED_DATA_DIR, "my_dataset.dataset"),
    output_directory=EMBEDDING_DIR,
    output_prefix="cell_embeddings",
    output_torch_embs=True,
)
print(f"Embeddings saved at: {EMBEDDING_DIR}")

# Plot embeddings UMAP
umap_plot_path = os.path.join(PLOT_DIR, "embeddings_umap")
embex.plot_embs(
    embs=embs_df,
    plot_style="umap",
    output_directory=PLOT_DIR,
    output_prefix="embeddings_umap",
    max_ncells_to_plot=max_cells_to_plot,
)
print(f"UMAP plot saved at: {umap_plot_path}")

# Rename the outputs directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.rename(BASE_DIR, f"{BASE_DIR}_{timestamp}")
print(f"Outputs directory renamed to: {BASE_DIR}_{timestamp}")
