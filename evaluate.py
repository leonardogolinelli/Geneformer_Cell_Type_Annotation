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
test_output_dir = os.path.join(RESULTS_DIR, "test_evaluation")
os.makedirs(test_output_dir, exist_ok=True)

# Load `id_class_dict` from the previously prepared dictionary
id_class_dict_path = os.path.join(RESULTS_DIR, "cell_classifier_id_class_dict.pkl")
with open(id_class_dict_path, "rb") as f:
    id_class_dict = pickle.load(f)
print("Loaded id_class_dict:", id_class_dict)

# Obtain the best checkpoint path programmatically
validation_dir = os.path.join(RESULTS_DIR, "validation")
best_checkpoint_dir = get_checkpoint_with_lowest_loss(validation_dir, datetime.datetime.now().strftime("%y%m%d"))
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
    output_prefix="test_metrics",
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
predictions_file = os.path.join(test_output_dir, "test_metrics_pred_dict.pkl")
predictions_plot_path = os.path.join(PLOT_DIR, "predictions_plot.png")

# Basically the heatmap of predictions, similar to the confusion matrix, but with
# granularity at the cell level
cc.plot_predictions(
    predictions_file=predictions_file,
    id_class_dict_file=id_class_dict_path,
    title="Cell Type Predictions",
    output_directory=PLOT_DIR,
    output_prefix="predictions_plot"
)
print(f"Predictions plot saved at: {predictions_plot_path}")
