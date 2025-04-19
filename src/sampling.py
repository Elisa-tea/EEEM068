# sampling.py

from abc import ABC, abstractmethod

class Sampling(ABC):  # parent class
    def __init__(self, video, clip_length):
        self.video = video
        self.clip_length = clip_length

    @abstractmethod
    def sample(self):
        pass

class EquidistantSampling(Sampling):
    def sample(self):
        # implement equidistant sampling across the entire video here
        pass

class InterpolationSampling(Sampling):
    def sample(self):
        # implement interpolation on short videos 
        pass

class Others(Sampling):
    def sample(self):
        # placeholder for other sampling methods
        pass