from src.model import load_model
from src.data import split_sources, process_dataset, train_augmentations
from transformers import TrainingArguments
from src.data import VideoDataCollator
from src.model import Metrics
from src.sampling import *
import argparse
import optuna
from optuna.pruners import MedianPruner
import torch
import numpy as np
import os
from functools import partial
import json
import pickle
import shutil


# Default configuration dictionary for notebook usage
DEFAULT_ARGS = {
    "train_dataset_path": "./HMDB_simp/",
    "val_dataset_path": "./HMDB_simp/",
    "model_type": "timesformer",  # options: "timesformer", "r3d"
    "n_trials": 20,
    "timeout": None,
    "study_name": "video_classification_study",
    "storage": None,
    "cache_dir": "./dataset_cache",
}


def get_sampler(sampler_type, **kwargs):
    clip_length = kwargs.get("clip_length", 8)
    if sampler_type == "fixed_step":
        return FixedStepSampler(step=kwargs.get("frame_step", clip_length))
    elif sampler_type == "equidistant":
        return EquidistantSampler(
            initial_offset=kwargs.get("initial_offset", 0),
            min_frames=kwargs.get("min_frames", clip_length),
        )
    elif sampler_type == "interpolation":
        return InterpolationSampler(min_frames=kwargs.get("min_frames", clip_length))
    elif sampler_type == "augmentation":
        return AugmentationSampler(min_frames=kwargs.get("min_frames", clip_length))
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


# Predefined sampler configurations
SAMPLER_CONFIGS = [
    # (sampler_type, clip_length, min_frames, frame_step, sample_rate)
    # Fixed step sampler configs
    ("fixed_step", 8, 8, 4, 0),     # Takes every 4th frame and uses 8 frames for each clip, skips videos with less than 8*4=32 frames
    ("fixed_step", 8, 8, 8, 0),     # Takes every 8th frame and uses 8 frames for each clip, skips videos with less than 8*8=64 frames  
    ("fixed_step", 16, 16, 4, 0),   # Takes every 4th frame and uses 16 frames for each clip, skips videos with less than 16*4=64 frames
    
    # Equidistant sampler configs
    ("equidistant", 8, 8, 0, 0.75),    # Equidistantly takes 75% frames and uses 8 frames for each clip
    ("equidistant", 8, 8, 0, 0.5),    # Equidistantly takes 50% frames and uses 8 frames for each clip
    ("equidistant", 8, 8, 0, 0.25),  # Equidistantly takes 25% frames and uses 8 frames for each clip
    ("equidistant", 16, 16, 0, 0.5),  # Equidistantly takes 50% frames and uses 16 frames for each clip
    
    # # Interpolation sampler configs
    # ("interpolation", 8, 8, 0, 0),  # Config 7
    
    # # Augmentation sampler configs
    # ("augmentation", 8, 8, 0, 0),   # Config 8
]


class DiskDatasetCache:
    """Cache for datasets that saves to disk instead of keeping in RAM"""
    
    def __init__(self, train_path, val_path, train_sources, val_sources, cache_dir="./dataset_cache"):
        self.train_path = train_path
        self.val_path = val_path
        self.train_sources = train_sources
        self.val_sources = val_sources
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a manifest file to keep track of cached datasets
        self.manifest_path = os.path.join(cache_dir, "manifest.json")
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}
    
    def get_dataset_key(self, use_augmentations, config_idx):
        """Generate a unique key for the dataset configuration"""
        return f"{int(use_augmentations)}_{config_idx}"
    
    def get_dataset_path(self, cache_key, is_train=True):
        """Get the path where a dataset is stored"""
        dataset_type = "train" if is_train else "val"
        return os.path.join(self.cache_dir, f"{cache_key}_{dataset_type}.pkl")
    
    def save_datasets(self, train_dataset, val_dataset, cache_key):
        """Save datasets to disk"""
        train_path = self.get_dataset_path(cache_key, is_train=True)
        val_path = self.get_dataset_path(cache_key, is_train=False)
        
        # Save datasets to disk
        with open(train_path, 'wb') as f:
            pickle.dump(train_dataset, f)
        
        with open(val_path, 'wb') as f:
            pickle.dump(val_dataset, f)
        
        # Update manifest
        self.manifest[cache_key] = {
            "train_path": train_path,
            "val_path": val_path,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset)
        }
        
        # Save manifest
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def load_datasets(self, cache_key):
        """Load datasets from disk"""
        train_path = self.get_dataset_path(cache_key, is_train=True)
        val_path = self.get_dataset_path(cache_key, is_train=False)
        
        # Load datasets from disk
        with open(train_path, 'rb') as f:
            train_dataset = pickle.load(f)
        
        with open(val_path, 'rb') as f:
            val_dataset = pickle.load(f)
        
        return train_dataset, val_dataset
    
    def get_datasets(self, use_augmentations, config_idx):
        """Get datasets with the given configuration, from disk cache if possible"""
        cache_key = self.get_dataset_key(use_augmentations, config_idx)
        
        if cache_key in self.manifest:
            print(f"Loading cached datasets from disk for configuration: {cache_key}")
            return self.load_datasets(cache_key)
        
        # Get the sampler configuration
        sampler_type, clip_length, min_frames, frame_step, sample_rate = SAMPLER_CONFIGS[config_idx]
        print(f"Creating new datasets for configuration: {cache_key} | {sampler_type}, clip_length={clip_length}, min_frames={min_frames}")
        
        # Get the appropriate sampler
        sampler = get_sampler(
            sampler_type,
            clip_length=clip_length,
            min_frames=min_frames,
            frame_step=frame_step,
            sample_rate=sample_rate
        )
        
        # Set up augmentation transform
        augmentation_transform = train_augmentations if use_augmentations else None
        
        # Process datasets
        train_dataset = process_dataset(
            self.train_path,
            self.train_sources,
            augmentation_transform=augmentation_transform,
            sampler=sampler,
            clip_length=clip_length,
        )
        
        val_dataset = process_dataset(
            self.val_path,
            self.val_sources,
            augmentation_transform=None,
            sampler=sampler,
            clip_length=clip_length,
        )
        
        # Save to disk cache
        self.save_datasets(train_dataset, val_dataset, cache_key)
        
        return train_dataset, val_dataset
    
    def clear_cache(self):
        """Clear the disk cache"""
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            self.manifest = {}
            
            # Re-create empty manifest
            with open(self.manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2)
            
            print(f"Cleared disk cache at {self.cache_dir}")


def objective(trial, dataset_cache, model_type):
    # Sample dataset hyperparameters
    use_augmentations = trial.suggest_categorical("use_augmentations", [True, False])
    config_idx = trial.suggest_int("sampler_config_idx", 0, len(SAMPLER_CONFIGS) - 1)
    
    # Get datasets based on the hyperparameters
    train_dataset, val_dataset = dataset_cache.get_datasets(use_augmentations, config_idx)
    
    # Sample model hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 2, 5)
    
    # Load model for each trial to start fresh
    extractor, model, device, TrainerClass = load_model(model_type)
    
    # Prepare data collator
    data_collator = VideoDataCollator(model_type=model_type)
    
    # Get metrics
    metrics = Metrics(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./optuna_output/trial_{trial.number}",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_dir=f"./logs/trial_{trial.number}",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to=None,
    )
    
    # Create trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics.compute_metrics,
        data_collator=data_collator,
    )
    
    # Only pass extractor if using transformers
    if model_type == "timesformer":
        trainer_kwargs["tokenizer"] = extractor
    
    trainer = TrainerClass(**trainer_kwargs)
    
    # Define callback for pruning
    class OptunaCallback:
        def __init__(self, trial):
            self.trial = trial
            self.step = 0
            
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            accuracy = metrics.get("eval_accuracy", 0)
            self.trial.report(accuracy, self.step)
            self.step += 1
            
            if self.trial.should_prune():
                raise optuna.TrialPruned()
    
    # Add callback
    trainer.add_callback(OptunaCallback(trial))
    
    # Train and evaluate
    trainer.train()
    
    # Get evaluation metrics
    final_metrics = trainer.evaluate()
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Return the metric to be optimized
    return final_metrics["eval_accuracy"]


def run_hyperparameter_optimization(args=None):
    """Run the hyperparameter optimization with given args or defaults"""
    # Use provided args or default args
    config = DEFAULT_ARGS.copy()
    if args is not None:
        config.update(args)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare paths
    TRAIN_DATASET_PATH = config["train_dataset_path"]
    VAL_DATASET_PATH = config["val_dataset_path"]
    
    # Split sources once for all trials
    train_sources, val_sources = split_sources(TRAIN_DATASET_PATH)
    
    # Initialize dataset cache
    dataset_cache = DiskDatasetCache(
        TRAIN_DATASET_PATH, 
        VAL_DATASET_PATH, 
        train_sources, 
        val_sources,
        cache_dir=config["cache_dir"]
    )
    
    # Create Optuna study
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(
        study_name=config["study_name"],
        storage=config["storage"],
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    
    # Create partial function with fixed arguments
    objective_func = partial(
        objective,
        dataset_cache=dataset_cache,
        model_type=config["model_type"],
    )
    
    # Optimize
    try:
        study.optimize(
            objective_func,
            n_trials=config["n_trials"],
            timeout=config["timeout"],
        )
    except KeyboardInterrupt:
        print("Optimization stopped manually.")
    
    # Print results
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Print best sampler configuration details
    if "sampler_config_idx" in best_trial.params:
        best_config_idx = best_trial.params["sampler_config_idx"]
        best_config = SAMPLER_CONFIGS[best_config_idx]
        print("\nBest sampler configuration:")
        print(f"  Sampler type: {best_config[0]}")
        print(f"  Clip length: {best_config[1]}")
        print(f"  Min frames: {best_config[2]}")
        print(f"  Frame step: {best_config[3]}")
        print(f"  Sample rate: {best_config[4]}")
    
    # Save best parameters to a JSON file
    with open("best_hyperparams.json", "w") as f:
        json.dump(best_trial.params, f, indent=2)
    
    print(f"Best hyperparameters saved to best_hyperparams.json")
    
    # Create a final model with the best parameters
    print("\nTraining final model with best parameters...")
    
    # Get datasets with the best configuration
    best_params = best_trial.params
    config_idx = best_params.get("sampler_config_idx", 0)
    use_augmentations = best_params.get("use_augmentations", False)
    
    train_dataset, val_dataset = dataset_cache.get_datasets(use_augmentations, config_idx)
    
    # Load model
    extractor, model, device, TrainerClass = load_model(config["model_type"])
    
    # Prepare data collator
    data_collator = VideoDataCollator(model_type=config["model_type"])
    
    # Get metrics
    metrics = Metrics(device)
    
    # Define training arguments with best parameters
    training_args = TrainingArguments(
        output_dir="./best_model_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=best_params.get("batch_size", 4),
        per_device_eval_batch_size=best_params.get("batch_size", 4),
        num_train_epochs=best_params.get("epochs", 3),
        learning_rate=best_params.get("learning_rate", 5e-5),
        weight_decay=best_params.get("weight_decay", 0.01),
        logging_dir="./logs/best_model",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to=None,
    )
    
    # Create trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics.compute_metrics,
        data_collator=data_collator,
    )
    
    # Only pass extractor if using transformers
    if config["model_type"] == "timesformer":
        trainer_kwargs["tokenizer"] = extractor
    
    trainer = TrainerClass(**trainer_kwargs)
    
    # Train final model
    trainer.train()
    
    # Evaluate final model
    final_metrics = trainer.evaluate()
    print("Final model performance:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value}")
    
    return {
        "study": study,
        "best_trial": best_trial,
        "best_params": best_trial.params,
        "final_metrics": final_metrics
    }


if __name__ == "__main__":
    # For command-line usage, parse arguments
    parser = argparse.ArgumentParser(description="Video classification hyperparameter tuning script")
    parser.add_argument("--train_dataset_path", type=str, default=DEFAULT_ARGS["train_dataset_path"])
    parser.add_argument("--val_dataset_path", type=str, default=DEFAULT_ARGS["val_dataset_path"])
    parser.add_argument("--model_type", type=str, default=DEFAULT_ARGS["model_type"], choices=["timesformer", "r3d"])
    parser.add_argument("--n_trials", type=int, default=DEFAULT_ARGS["n_trials"])
    parser.add_argument("--timeout", type=int, default=DEFAULT_ARGS["timeout"])
    parser.add_argument("--study_name", type=str, default=DEFAULT_ARGS["study_name"])
    parser.add_argument("--storage", type=str, default=DEFAULT_ARGS["storage"])
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_ARGS["cache_dir"])
    
    args = parser.parse_args()
    
    # Convert namespace to dictionary
    args_dict = vars(args)
    
    # Run optimization
    run_hyperparameter_optimization(args_dict) 