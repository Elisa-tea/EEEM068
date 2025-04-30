from abc import ABC, abstractmethod
import os
import torch
import random
import torch.nn.functional as F


class Sampler(ABC):
    @abstractmethod
    def sample(self, frame_dir=None, *args, **kwargs):
        pass

    @staticmethod
    def list_frames(frame_dir):
        return [
            os.path.join(frame_dir, file)
            for file in sorted(os.listdir(frame_dir))
            if file.endswith((".jpg", ".png", ".jpeg"))
        ]


class FixedStepSampler(Sampler):
    def __init__(self, step=8):
        self.step = step
        
    def sample(self, frame_dir):
        """
        Load every [step]-th frame from a directory.
        """
        frame_files = self.list_frames(frame_dir)
        return frame_files[::self.step]


class EquidistantSampler(Sampler):
    def __init__(self, initial_offset=5, min_frames=8):
        self.initial_offset = initial_offset
        self.min_frames = min_frames
        
    def sample(self, frame_dir):
        frame_files = self.list_frames(frame_dir)
        total_frames = len(frame_files)
        
        if total_frames <= self.initial_offset:
            return frame_files  # Not enough frames, return all

        step = max(1, int((total_frames - self.initial_offset) / self.min_frames))
        
        return frame_files[self.initial_offset::step]

class InterpolationSampler(Sampler):
    """
    Sample frames from a video by interpolating between key frames.
    """
    def __init__(self, min_frames=8):
        self.min_frames = min_frames
        
    def sample(self, frame_dir):
        # TODO
        frame_files = self.list_frames(frame_dir)
        return frame_files  # Placeholder implementation
        
class AugmentationSampler(Sampler):
    """
    Sample frames from a video by adding new augmented frames.
    """
    def __init__(self, min_frames=8):
        self.min_frames = min_frames
    
    def random_horizontal_flip(frame, p=0.8):
        """
        #this highly preserve the content of the image#
        Apply random horizontal flip to an image frame.

        Args:
            frame (Tensor): Image tensor of shape (C, H, W).
            p (float): Probability of applying the flip.

        Returns:
            Tensor: Horizontally flipped image tensor of shape (C, H, W) if flipped, else the original.
        """
        if random.random() < p:  # Flip with probability p
            return F.hflip(frame)
        return frame
    

    def vertical_down_translation(frame, shift=20):
        """
        Apply vertical down translation to an image frame.

        Args:
            frame (Tensor): Image tensor of shape (C, H, W).
            shift (int): Number of pixels to shift the image downward.

        Returns:
            Tensor: Translated image tensor of shape (C, H, W).
        """
        C, H, W = frame.shape  # Get channel, height, width

        # Create a black canvas (zero tensor)
        translated_frame = torch.zeros_like(frame)

        # Shift the original image down, filling the top with black pixels
        if shift < H:  # Ensure shift is within bounds
            translated_frame[:, shift:, :] = frame[:, :-shift, :]

        return translated_frame

    def sample(self, frame_dir, augmentation_type=random_horizontal_flip, min_frames=8, sample_rate=32):
        """
        Creates a frames_files from a video tensor based on the length conditions.

        Args:
            video (Tensor): Video tensor of shape (num_frames, C, H, W).
            min_frames (int): Number of frames in the final frames_files.
            sample_rate (int): Interval between sampled frames.

        Returns:
            Tensor: A single frames_files tensor of shape (min_frames, C, H, W).
        """

        video = torch.load(frame_dir)  # Load the video tensor
        
        num_frames = video.shape[0]
        print(f"Total frames in video: {num_frames}")

        if num_frames >= min_frames * sample_rate:
            # Sample every 32 frames to create an 8-frame frames_files
            frames_files = video[::sample_rate][:min_frames]

        else:
            # Sample available frames at the given sample_rate
            n = num_frames // sample_rate  # Compute how many frames we can sample
            n_frames_files = video[::sample_rate][:n]
            remaining_frames_needed = min_frames - n

            # Initialize additional_frames_files with an empty list to store frames
            additional_frames = []

            # Start sampling additional frames
            start_idx = 0
            while len(additional_frames) < remaining_frames_needed:
                idx = (start_idx + sample_rate) % num_frames  # circular
                t = video[idx:idx + 1]
                additional_frames.append(t)
                start_idx += sample_rate

            # Convert list to tensor and apply augmentation
            additional_frames_files = torch.cat(additional_frames, dim=0)
            additional_frames_files = torch.stack([augmentation_type(f) for f in additional_frames_files])

            # Concatenate the sampled frames_files with the additional frames
            frames_files = torch.cat([n_frames_files, additional_frames_files], dim=0)

        return frames_files
       