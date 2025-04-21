from abc import ABC, abstractmethod

class Sampling(ABC):  # parent class
    def __init__(self, video, clip_length):
        self.video = video
        self.clip_length = clip_length

    @abstractmethod
    def sample(self):
        pass

class EquidistantSampling(Sampling):
    def __init__(self, video, clip_length):
        super().__init__(video, clip_length)

    def sample(self):
        total_frames = len(self.video)
        if self.clip_length >= total_frames or total_frames == 0:
            raise ValueError('clip_length longer than some videos!reduce it!')   # raise error for now, can modify later

        offset = 5  # number of frames to skip at the start, hard-coded for now, can change it to variable later
        step = (total_frames - offset) / self.clip_length # (total_frames - offset) is the actual number of frames to sample from
        indices = [int(offset + i * step) for i in range(self.clip_length)]

class InterpolationSampling(Sampling):
    def __init__(self, video, clip_length):
        super().__init__(video, clip_length)

    def sample(self):
        # implement interpolation on short videos 
        pass

class SamplingWithAug(Sampling):
    def __init__(self, video, clip_length=8, sample_rate=32, augmentation_type="random_horizontal_flip"):
        super().__init__(video, clip_length)
        self.sample_rate = sample_rate
        self.augmentation_type = augmentation_type

    def sample(self):
        # implement augmented sampling here
        pass

class Others(Sampling):
    def __init__(self, video, clip_length):
        super().__init__(video, clip_length)

    def sample(self):
        # placeholder for other sampling methods
        pass