from abc import ABC, abstractmethod
import os
import cv2
import numpy as np


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

class FrameSampler(Sampler):
    #sub-parent class for selecting frame locations
    @abstractmethod
    def sample(self, frame_dir):
        pass

class ClipSampler(Sampler):
    #sub-parent class for creating frames
    @abstractmethod
    def sample(self, frame_dir):
        pass


class FixedStepSampler(FrameSampler):
    def __init__(self, step=8):
        self.step = step
        
    def sample(self, frame_dir):
        """
        Load every [step]-th frame from a directory.
        """
        frame_files = self.list_frames(frame_dir)
        return frame_files[::self.step]


class EquidistantSampler(FrameSampler):
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


class InterpolationSampler(ClipSampler):
    """
    Sample frames from a video by interpolating between key frames.
    Outputs interpolated frames as a numpy array (and frame positions for checking purposes).
    """
    def __init__(self, min_frames=8):
        self.min_frames = min_frames
        self.transform = transforms.ToTensor()
        
    def sample(self, frame_dir):
        frame_files = self.list_frames(frame_dir)
        total_frames = len(frame_files)

        if total_frames == 0:
            raise ValueError("Video is empty")
        
        else:
            video = []
            for f in frame_files:
                frame = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                video.append(frame)
                
            positions = np.linspace(0, total_frames - 1, self.min_frames)
            clip = []
            for pos in positions:
                low_idx = int(np.floor(pos))
                high_idx = min(low_idx + 1, total_frames - 1)
                alpha = pos - low_idx

                frame_low = video[low_idx]
                frame_high = video[high_idx]
                
                interp_frame = (1 - alpha) * frame_low + alpha * frame_high
                interp_frame = np.clip(interp_frame, 0, 255).astype(np.uint8)       
                clip.append(interp_frame)
                
        clip_frames = np.stack(clip, axis=0)
        positions_frames = list(positions)
        
        return clip_frames, positions_frames 
        
class AugmentationSampler(ClipSampler):
    """
    Sample frames from a video by adding new augmented frames.
    """
    def __init__(self, min_frames=8):
        self.min_frames = min_frames
        
    def sample(self, frame_dir):
        # TODO: Elisa-tea
        frame_files = self.list_frames(frame_dir)
        return frame_files  # Placeholder implementation
