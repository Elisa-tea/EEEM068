import os
import torch
from tqdm import tqdm
import random
from .sampling import Sampler, FixedStepSampler
from .augmentations import default_transforms, train_augmentations

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
    def __init__(self, model_type="timesformer"):
        self.model_type = model_type

    def __call__(
        self, features: list[tuple[torch.Tensor, int]]
    ) -> dict[str, torch.Tensor]:
        clips, labels = zip(*features)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        if self.model_type == "r3d":
            clips = [clip.permute(1, 0, 2, 3) for clip in clips]

        video_tensor = torch.stack(clips)

        if self.model_type == "timesformer":
            return {"pixel_values": video_tensor, "labels": label_tensor}
        elif self.model_type == "r3d":
            return {"input": video_tensor, "labels": label_tensor}
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


def split_sources(
    dataset_path, train_ratio=0.8
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
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
        train_sources[category] = instances[
            :split_idx
        ]  # First [train_ratio] for training
        val_sources[category] = instances[
            split_idx:
        ]  # Last [1-train_ratio] for validation

    return train_sources, val_sources


def create_clips(frames, clip_size=8) -> list[torch.Tensor]:
    """
    Given a list of sampled frames, create multiple [clip_size]-frame clips.
    Each clip is returned as a tensor.
    """
    clips = []

    for i in range(0, len(frames) - clip_size + 1, clip_size):
        clip = frames[i : i + clip_size]
        if len(clip) == clip_size:
            clips.append(torch.stack(clip))  # Convert the clip to a tensor
    return clips


def process_dataset(
    dataset_path,
    sources_dict,
    augmentation_transform=None,
    sampler: Sampler = FixedStepSampler(),
    clip_length: int = 8,
) -> list[tuple[torch.Tensor, int]]:
    """
    Processes dataset based on a predefined list of sources.
    """
    if augmentation_transform is None:
        augmentation_transform = lambda image: {"image": image}

    dataset = []

    for category, instances in tqdm(sources_dict.items()):
        category_path = os.path.join(dataset_path, category)

        for instance in instances:
            instance_path = os.path.join(category_path, instance)
            if not os.path.isdir(instance_path):
                continue

            frames = sampler.sample(instance_path)

            for i, frame in enumerate(frames):
                try:
                    frames[i] = default_transforms(
                        image=augmentation_transform(image=frame)["image"]
                    )["image"]

                except Exception as e:
                    print(f"Error processing frame {i}: {e}")
                    continue

            clips = create_clips(frames, clip_length)
            for clip in clips:
                dataset.append((clip, CATEGORY_INDEX[category]))

    return dataset


if __name__ == "__main__":
    DATASET_PATH = "HMDB_simp/"
    train_sources, val_sources = split_sources(DATASET_PATH)

    train_dataset = process_dataset(
        DATASET_PATH, train_sources, augmentation_transform=train_augmentations
    )