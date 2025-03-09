from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEFeatureExtractor
import torchvision.transforms as transforms
import torch
import os
import cv2  # OpenCV for video loading
