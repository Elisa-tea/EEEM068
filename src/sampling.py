from abc import ABC, abstractmethod
import os


class Sampler(ABC):
    @staticmethod
    @abstractmethod
    def sample(frame_dir=None, *args, **kwargs):
        pass

    @staticmethod
    def list_frames(frame_dir):
        return [
            os.path.join(frame_dir, file)
            for file in sorted(os.listdir(frame_dir))
            if file.endswith((".jpg", ".png", ".jpeg"))
        ]


class FixedStepSampler(Sampler):
    @staticmethod
    def sample(frame_dir, frame_rate=8):
        """
        Load every [frame_rate]-th frame from a directory and apply transformations.
        """
        frame_files = Sampler.list_frames(frame_dir)
        return frame_files[::frame_rate]


class EquidistantSampler(Sampler):
    @staticmethod
    def sample(frame_dir, initial_offset=5, min_frames=8):
        frame_files = Sampler.list_frames(frame_dir)
        total_frames = len(frame_files)

        step = (total_frames - initial_offset) / min_frames

        return frame_files[initial_offset::step]

class InterpolationSampler(Sampler):
    """
    Sample frames from a video by interpolating between key frames.
    """
    @staticmethod
    def sample(frame_dir, min_frames=8):
        # TODO
        pass
        
class AugmentationSampler(Sampler):
    """
    Sample frames from a video by adding new augmented frames.
    """
    @staticmethod
    def sample(frame_dir, min_frames=8):
        # TODO: Elisa-tea
        pass
