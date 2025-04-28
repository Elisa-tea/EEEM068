import albumentations as A
import albumentations.pytorch

train_augmentations = A.Compose(
    [
        A.Resize(224, 224),
        # Brightness/Contrast
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            ],
            p=0.4,
        ),
        # Noise/Quality
        A.OneOf(
            [
                A.GaussNoise(p=0.5),
                A.ImageCompression(quality_range=(65, 95), p=0.5),
            ],
            p=0.4,
        ),
    ]
)

default_transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ]
)
