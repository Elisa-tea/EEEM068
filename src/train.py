from model import load_pretrained_vit
from data import split_sources, process_dataset, train_augmentations
from transformers import Trainer, TrainingArguments
from data import VideoDataCollator
from model import Metrics

if __name__ == "__main__":
    extractor, model, device = load_pretrained_vit()

    DATASET_PATH = "../HMDB_simp/"

    train_sources, val_sources = split_sources(DATASET_PATH)

    train_dataset = process_dataset(
        DATASET_PATH, train_sources, augmentation_transform=train_augmentations
    )

    val_dataset = process_dataset(
        DATASET_PATH, val_sources, augmentation_transform=None
    )

    dataset_size = len(train_dataset) + len(val_dataset)

    data_collator = VideoDataCollator()

    print(
        f"Total clips: {dataset_size}, Train: {len(train_dataset)}, Val: {len(val_dataset)}"
    )

    metrics = Metrics(device)

    training_args = TrainingArguments(
        output_dir="./timesformer_output",  # Save checkpoints
        eval_strategy="epoch",  # Evaluate after every epoch
        save_strategy="epoch",  # Save model after each epoch
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",  # TensorBoard logs
        logging_steps=10,
        save_total_limit=2,  # Keep only last 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=extractor,  # Feature extractor
        compute_metrics=metrics.compute_metrics,
        data_collator=data_collator,
    )
    
    trainer.train()
