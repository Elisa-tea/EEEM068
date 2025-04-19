# sampling.py

class Sampling: #parent class
    def __init__(self, video, num_frames):
        self.video = video
        self.num_frames = num_frames

    def sample(self):
        raise NotImplementedError("Each subclass must implement this method.")

class EquidistantSampling(Sampling):
    def sample(self):
        # implement equidistant sampling across the eentire video here
        pass

class InterpolationSampling(Sampling):
    def sample(self):
        # implement interpolation on short videos 
        pass

class Others(Sampling):
    def sample(self):
        # placeholder for other sampling methods
        pass