from src.model import load_model
from src.data import split_sources, process_dataset, train_augmentations
from transformers import TrainingArguments
from src.data import VideoDataCollator
from src.model import Metrics
from src.sampling import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Video classification training script")
    parser.add_argument(
        "--use_augmentations",
        action="store_true",
        help="Whether to use data augmentations",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="fixed_step",
        choices=["fixed_step", "equidistant", "interpolation", "augmentation"],
        help="Sampler to use for frame selection",
    )
    parser.add_argument(
        "--frame_step", type=int, default=8, help="Step size for fixed step sampler"
    )
    parser.add_argument(
        "--sample_rate", type=float, default=1, help="Sample rate for equidistant sampler"
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=8,
        help="Minimum number of frames expected from sampler",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=8,
        help="How many frames are used for each clip",
    )
    parser.add_argument(
        "--initial_offset",
        type=int,
        default=5,
        help="Initial offset for equidistant sampler",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/HMDB_simp/",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default="/HMDB_simp/",
        help="Path to validation dataset",
    )
    # Training arguments
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Per device training batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Per device evaluation batch size",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="timesformer",
        choices=["timesformer", "r3d"],
        help="Model to use for video classification",
    )

    return parser.parse_args()


def get_sampler(sampler_type, **kwargs):
    clip_length = kwargs.get("clip_length", 8)
    if sampler_type == "fixed_step":
        return FixedStepSampler(step=kwargs.get("frame_step", clip_length))
    elif sampler_type == "equidistant":
        return EquidistantSampler(
            sample_rate=kwargs.get("sample_rate", 1),
            min_frames=kwargs.get("min_frames", clip_length),
        )
    elif sampler_type == "interpolation":
        return InterpolationSampler(min_frames=kwargs.get("min_frames", clip_length))
    elif sampler_type == "augmentation":
        return AugmentationSampler(min_frames=kwargs.get("min_frames", clip_length))
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


if __name__ == "__main__":
    args = parse_args()
    extractor, model, device, TrainerClass = load_model(args.model_type)

    TRAIN_DATASET_PATH = args.train_dataset_path
    VAL_DATASET_PATH = args.val_dataset_path

    train_sources, val_sources = split_sources(TRAIN_DATASET_PATH)

    sampler = get_sampler(
        args.sampler,
        clip_length=args.clip_length,
        frame_step=args.frame_step,
        sample_rate=args.sample_rate,
        min_frames=args.min_frames
    )

    augmentation_transform = train_augmentations if args.use_augmentations else None

    train_dataset = process_dataset(
        TRAIN_DATASET_PATH,
        train_sources,
        augmentation_transform=augmentation_transform,
        sampler=sampler,
        clip_length=args.clip_length,
    )

    val_dataset = process_dataset(
        VAL_DATASET_PATH, val_sources, augmentation_transform=None, sampler=sampler,
        clip_length=args.clip_length,
    )

    dataset_size = len(train_dataset) + len(val_dataset)

    data_collator = VideoDataCollator(model_type=args.model_type)

    print(
        f"Total clips: {dataset_size}, Train: {len(train_dataset)}, Val: {len(val_dataset)}"
    )

    metrics = Metrics(device)

    training_args = TrainingArguments(
        output_dir="./timesformer_output",  # Save checkpoints
        eval_strategy="epoch",  # Evaluate after every epoch
        save_strategy="epoch",  # Save model after each epoch
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_dir="./logs",  # TensorBoard logs
        logging_steps=10,
        save_total_limit=2,  # Keep only last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to=None,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metrics.compute_metrics,
        data_collator=data_collator,
    )

    # Only pass extractor if using transformers
    if args.model_type == "timesformer":
        trainer_kwargs["tokenizer"] = extractor

    trainer = TrainerClass(**trainer_kwargs)

    trainer.train()
