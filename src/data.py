from abc import ABC, abstractmethod
import os
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import random
from sampling import Sampler, FixedStepSampler
from augmentations import default_transforms, train_augmentations

CATEGORY_INDEX = {
    "brush_hair": 0,
    "cartwheel": 1,
    "catch": 2,
    "chew": 3,
    "climb": 4,
    "climb_stairs": 5,
    "draw_sword": 6,
    "eat": 7,
    "fencing": 8,
    "flic_flac": 9,
    "golf": 10,
    "handstand": 11,
    "kiss": 12,
    "pick": 13,
    "pour": 14,
    "pullup": 15,
    "pushup": 16,
    "ride_bike": 17,
    "shoot_bow": 18,
    "shoot_gun": 19,
    "situp": 20,
    "smile": 21,
    "smoke": 22,
    "throw": 23,
    "wave": 24,
}


class VideoDataCollator:
    """
    Custom data collator for TimeSFormer.
    Converts (clip, label) tuples into a dictionary format.
    """

    def __call__(self, features):
        clips, labels = zip(*features)  # Unpack (clip, label)
        batch = {
            "pixel_values": torch.stack(clips),  # Stack clips into batch
            "labels": torch.tensor(
                labels, dtype=torch.long
            ),  # Convert labels to tensor
        }
        return batch


def split_sources(dataset_path, train_ratio=0.8):
    """
    Splits source folders into train and val sets before processing clips.
    Ensures that all clips from a source video stay in the same set.
    """
    train_sources = {}
    val_sources = {}

    for category in os.listdir(dataset_path):  # Iterate over action categories
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue

        instances = os.listdir(category_path)  # List all source folders (video IDs)
        random.shuffle(instances)  # Shuffle instances before splitting

        split_idx = int(len(instances) * train_ratio)
        train_sources[category] = instances[:split_idx]  # First 80% for training
        val_sources[category] = instances[split_idx:]  # Last 20% for validation

    return train_sources, val_sources


def create_clips(frames, clip_size=8):
    """
    Given a list of sampled frames, create multiple [clip_size]-frame clips.
    """
    clips = []

    for i in range(0, len(frames) - clip_size + 1, clip_size):
        clip = frames[i : i + clip_size]
        if len(clip) == clip_size:
            clips.append(clip)
    return clips


def process_dataset(
    dataset_path,
    sources_dict,
    augmentation_transform=None,
    sampler: Sampler = FixedStepSampler,
):
    """
    Processes dataset based on a predefined list of sources.
    """
    if augmentation_transform is None:
        augmentation_transform = lambda x: {"image": x}

    dataset = []

    for category, instances in tqdm(sources_dict.items()):
        category_path = os.path.join(dataset_path, category)

        for instance in instances:
            instance_path = os.path.join(category_path, instance)
            if not os.path.isdir(instance_path):
                # print(f"Skipping non-directory file: {instance_path}")
                continue

            # Load sampled frames
            frame_paths = sampler.sample(instance_path)

            frames = []

            for path in frame_paths:
                try:
                    frames.append(
                        default_transforms(
                            image=augmentation_transform(
                                image=cv2.cvtColor(
                                    cv2.imread(path), cv2.COLOR_BGR2RGB
                                )
                            )["image"]
                        )["image"]
                    )
                except Exception as e:
                    print(f"Error processing frame {path}: {e}")
                    frames.append(None)

            # Create 8-frame clips
            clips = create_clips(frames, 8)
            for clip in clips:
                dataset.append((clip, CATEGORY_INDEX[category]))

    return dataset  # List of (clip, label)


if __name__ == "__main__":
    DATASET_PATH = "HMDB_simp/"
    # print(len(FixedStepSampler.sample(frame_dir="../HMDB_simp/")))
    train_sources, val_sources = split_sources(DATASET_PATH)

    # Pr`ocess train and val sets separately
    train_dataset = process_dataset(
        DATASET_PATH, train_sources, augmentation_transform=train_augmentations
    )
    a = train_dataset[0]
