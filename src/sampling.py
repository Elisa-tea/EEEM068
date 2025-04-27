from abc import ABC, abstractmethod
import os


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
        
    def sample(self, frame_dir):
        # TODO: Elisa-tea
        frame_files = self.list_frames(frame_dir)
        return frame_files  # Placeholder implementation
