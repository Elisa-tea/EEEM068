from abc import ABC, abstractmethod
import os
import torch.nn.functional as F
import cv2
import numpy as np
from .augmentations import sampling_augmentations
import random


class Sampler(ABC):
    """
    Abstract base class for all samplers.
    """

    @abstractmethod
    def sample(self, frame_dir=None, *args, **kwargs) -> list[np.ndarray]:
        """
        Abstract method to sample frames from a video.
        """
        pass

    @staticmethod
    def list_frames(frame_dir) -> list[str]:
        """
        List all frames in a directory.
        """
        return [
            os.path.join(frame_dir, file)
            for file in sorted(os.listdir(frame_dir))
            if file.endswith((".jpg", ".png", ".jpeg"))
        ]

    @staticmethod
    def read_frames(frame_files) -> list[np.ndarray]:
        """
        Read all frames from a list of file paths.
        """
        return [
            cv2.cvtColor(cv2.imread(frame_file), cv2.COLOR_BGR2RGB)
            for frame_file in frame_files
        ]


class FixedStepSampler(Sampler):
    def __init__(self, step=8):
        self.step = step

    def sample(self, frame_dir):
        """
        Load every [step]-th frame from a directory.
        """
        frame_files = self.list_frames(frame_dir)
        return self.read_frames(frame_files[:: self.step])


class EquidistantSampler(Sampler):
    def __init__(self, min_frames=8, sample_rate=1):
        self.min_frames = min_frames
        self.sample_rate = sample_rate

    def sample(self, frame_dir):
        """
        Sample frames from a video by selecting equidistant frames.
        """
        frame_files = self.list_frames(frame_dir)
        total_frames = len(frame_files)
        frames_to_sample = max(self.min_frames, int(total_frames * self.sample_rate))

        if total_frames <= frames_to_sample:
            return self.read_frames(frame_files)

        step = max(1, int(total_frames / frames_to_sample))

        return self.read_frames(frame_files[::step])


class InterpolationSampler(Sampler):
    """
    Sample frames from a video by interpolating between key frames.
    Outputs interpolated frames as a numpy array (and frame positions for checking purposes).
    """

    def __init__(self, min_frames=8):
        self.min_frames = min_frames

    def sample(self, frame_dir):
        """
        Sample frames from a video by interpolating between key frames.
        """
        input_frames = self.read_frames(self.list_frames(frame_dir))
        total_frames = len(input_frames)

        if total_frames <= 1 or total_frames >= self.min_frames:
            return input_frames

        frames_to_fill = self.min_frames - total_frames
        positions = np.linspace(0, total_frames - 1, frames_to_fill)

        output_frames = [(i, frame) for i, frame in enumerate(input_frames)]
        for pos in positions:
            low_idx = int(np.floor(pos))
            high_idx = min(low_idx + 1, total_frames - 1)
            alpha = pos - low_idx

            frame_low = input_frames[low_idx]
            frame_high = input_frames[high_idx]

            interp_frame = cv2.addWeighted(frame_low, 1 - alpha, frame_high, alpha, 0)
            output_frames.append((pos, interp_frame))

        output_frames = [
            frame for _, frame in sorted(output_frames, key=lambda x: x[0])
        ]
        return output_frames


class AugmentationSampler(Sampler):
    """
    Sample frames from a video by adding new augmented frames.
    """

    def __init__(self, min_frames=8, augmentations=sampling_augmentations):
        self.min_frames = min_frames
        self.augmentations = augmentations

    def sample(self, frame_dir) -> list[np.ndarray]:
        """
        Creates augmented frames from existing frames if needed to reach min_frames.
        Preserves frame order by tracking positions.
        """

        input_frames = self.read_frames(self.list_frames(frame_dir))
        total_frames = len(input_frames)

        if total_frames == 0 or total_frames >= self.min_frames:
            return input_frames

        frames_to_add = self.min_frames - total_frames

        output_frames = [(i, frame) for i, frame in enumerate(input_frames)]

        positions = np.linspace(0, total_frames - 1, frames_to_add)

        for pos in positions:

            source_idx = int(np.floor(pos))

            source_frame = input_frames[source_idx]

            augmented = self.augmentations(image=source_frame)["image"]

            output_frames.append((source_idx + random.uniform(0.0, 0.1), augmented))

        output_frames = [
            frame for _, frame in sorted(output_frames, key=lambda x: x[0])
        ]

        return output_frames
