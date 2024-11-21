import os
import datetime
import pickle
import numpy as np
import pandas as pd
from geneformer import Classifier, EmbExtractor
from utils import plot_confusion, get_checkpoint_with_lowest_loss  # Ensure these functions are correctly defined
import torch
import gc

# Define the base directory

BASE_DIR = os.path.abspath("outputs")  # Base directory for outputs
PLOT_DIR = os.path.join(BASE_DIR, "plots")  # Directory for plots
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")  # Directory for embeddings
TOKENIZED_DATA_DIR = os.path.join(BASE_DIR, "tokenized_data")  # Directory for tokenized data
RESULTS_DIR = os.path.join(BASE_DIR, "files")  # Directory for files
test_output_dir = os.path.join(RESULTS_DIR, "test_evaluation")  # Test evaluation output directory

# Load `id_class_dict` from the previously prepared dictionary
id_class_dict_path = os.path.join(RESULTS_DIR, "cell_classifier_id_class_dict.pkl")
with open(id_class_dict_path, "rb") as f:
    id_class_dict = pickle.load(f)
print("Loaded id_class_dict:", id_class_dict)

# Obtain the best checkpoint path programmatically
validation_dir = os.path.join(RESULTS_DIR, "validation")
current_date = datetime.datetime.now().strftime("%y%m%d")
best_checkpoint_dir = get_checkpoint_with_lowest_loss(validation_dir, current_date)
best_checkpoint_path = os.path.abspath(best_checkpoint_dir)
print("Best checkpoint path:", best_checkpoint_path)

# Initialize the Classifier instance
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "cell_types", "states": "all"},
    forward_batch_size=10,  # Adjust for memory efficiency
    nproc=10,
)

# Clear memory to prevent potential issues
torch.cuda.empty_cache()
gc.collect()

# Perform model evaluation
test_metrics = cc.evaluate_saved_model(
    model_directory=best_checkpoint_path,
    id_class_dict_file=id_class_dict_path,
    test_data_file=os.path.join(RESULTS_DIR, "cell_classifier_labeled_test.dataset"),
    output_directory=test_output_dir,
    output_prefix="eval_loss",
    predict=True,
)

# Define the path for the stats file
stats_file_path = os.path.join(BASE_DIR, "metrics.txt")

# Write Test Metrics to a text file
with open(stats_file_path, "w") as stats_file:
    stats_file.write("Test Evaluation Metrics:\n")
    for metric, value in test_metrics.items():
        if isinstance(value, (list, np.ndarray, pd.DataFrame)):
            stats_file.write(f"{metric}:\n{value}\n\n")
        else:
            stats_file.write(f"{metric}: {value}\n")

# Save the confusion matrix
confusion_matrix_plot_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
plot_confusion(test_metrics["conf_matrix"], save_path=confusion_matrix_plot_path)
print(f"Confusion matrix saved at: {confusion_matrix_plot_path}")

# Visualize Predictions
predictions_file = os.path.join(test_output_dir, "test_model_pred_dict.pkl")
predictions_plot_path = os.path.join(PLOT_DIR, "predictions_plot.png")

cc.plot_predictions(
    predictions_file=predictions_file,
    id_class_dict_file=id_class_dict_path,
    title="Cell Type Predictions",
    output_directory=PLOT_DIR,
    output_prefix="predictions_plot"
)
print(f"Predictions plot saved at: {predictions_plot_path}")

# Load id_class_dict for EmbExtractor
print("Loaded id_class_dict for EmbExtractor:", id_class_dict)

# Get unique cell types
cell_types_unique = list(id_class_dict.values())

# Initialize EmbExtractor
embex = EmbExtractor(
    model_type="CellClassifier",
    num_classes=len(id_class_dict),
    emb_mode="cell",
    filter_data={"cell_types": cell_types_unique},
    max_ncells=2000,
    emb_layer=-1,
    emb_label=["cell_types"],
    labels_to_plot=["cell_types"],
    forward_batch_size=10,  # Adjust for memory efficiency
    nproc=10,  # Number of processes
)

# Clear memory before embedding extraction
torch.cuda.empty_cache()
gc.collect()

# Extract embeddings
embs_df, embs_tensor = embex.extract_embs(
    model_directory=best_checkpoint_path,
    input_data_file=os.path.join(TOKENIZED_DATA_DIR, "my_dataset.dataset"),
    output_directory=EMBEDDING_DIR,
    output_prefix="cell_embeddings",
    output_torch_embs=True,
)
print(f"Embeddings saved at: {EMBEDDING_DIR}")

# UMAP Plot
umap_plot_path = os.path.join(PLOT_DIR, "embeddings_umap")
embex.plot_embs(
    embs=embs_df,
    plot_style="umap",
    output_directory=PLOT_DIR,
    output_prefix="embeddings_umap",
    max_ncells_to_plot=2000,
)
print(f"UMAP plot saved at: {umap_plot_path}")

# Heatmap Plot
heatmap_plot_path = os.path.join(PLOT_DIR, "embeddings_heatmap")
embex.plot_embs(
    embs=embs_df,
    plot_style="heatmap",
    output_directory=PLOT_DIR,
    output_prefix="embeddings_heatmap",
    max_ncells_to_plot=None
)
print(f"Heatmap plot saved at: {heatmap_plot_path}")

# Rename the outputs directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.rename(BASE_DIR, f"{BASE_DIR}_{timestamp}")
print(f"Outputs directory renamed to: {BASE_DIR}_{timestamp}")
