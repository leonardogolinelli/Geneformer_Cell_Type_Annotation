import os
import datetime
import pickle
from geneformer import Classifier
from utils import get_checkpoint_with_lowest_loss

# Inputs
MODEL_NAME = "gf-6L-30M-i2048"
BASE_DIR = os.path.abspath("outputs")  # Base directory for outputs
RESULTS_DIR = os.path.join(BASE_DIR, "files")
MODEL_DIR = os.path.abspath(f"pretrained_models/{MODEL_NAME}")
TOKENIZED_DATA_DIR = os.path.join(BASE_DIR, "tokenized_data")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
id_class_dict_path = os.path.join(RESULTS_DIR, "cell_classifier_id_class_dict.pkl")
validation_dir = os.path.join(RESULTS_DIR, "validation")
os.makedirs(validation_dir, exist_ok=True)

# Hyperparameters
training_args = {
    "num_train_epochs": 1,
    "learning_rate": 0.000804,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 1812,
    "weight_decay": 0.258828,
    "per_device_train_batch_size": 12,
    "seed": 73,
}

# Initialize the Classifier
cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "cell_types", "states": "all"},
    filter_data=None,
    training_args=training_args,  # Empty since not training
    forward_batch_size=10,
    freeze_layers=10,
    num_crossval_splits=1,
    nproc=20,
)

# Load relevant dictionaries

split_dict_path = os.path.join(RESULTS_DIR, "train_test_id_split_dict.pkl")
with open(split_dict_path, "rb") as f:
    train_test_id_split_dict = pickle.load(f)

split_dict_path = os.path.join(RESULTS_DIR, "train_eval_id_split_dict.pkl")
with open(split_dict_path, "rb") as f:
    train_eval_id_split_dict = pickle.load(f)

# Prepare data for training and evaluation
cc.prepare_data(
    input_data_file=os.path.join(TOKENIZED_DATA_DIR, "my_dataset.dataset"),
    output_directory=RESULTS_DIR,
    output_prefix="cell_classifier",
    split_id_dict=train_test_id_split_dict,
)

# Load the id_class_dict generated during data preparation
id_class_dict_path = os.path.join(RESULTS_DIR, "cell_classifier_id_class_dict.pkl")
with open(id_class_dict_path, "rb") as f:
    id_class_dict = pickle.load(f)
print("Loaded id_class_dict:", id_class_dict)

all_metrics = cc.validate(
    model_directory=MODEL_DIR,
    prepared_input_data_file=os.path.join(RESULTS_DIR, "cell_classifier_labeled_train.dataset"),
    id_class_dict_file=id_class_dict_path,  # Use the id_class_dict from prepare_data
    output_directory=validation_dir,
    output_prefix="fine_tuned_model",
    split_id_dict=train_eval_id_split_dict,
    n_hyperopt_trials=0,
)

print("Validation complete. Metrics:", all_metrics)

