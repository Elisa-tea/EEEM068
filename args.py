# args.py  ##placeholder and template for now, will modify the actual code for based on our case
import argparse  

def get_args():
    parser = argparse.ArgumentParser(description="Training setup for ViT project")

    parser.add_argument('--strategy', type=str, default='equidistant',
                        choices=['equidistant', 'interpolation', 'augmented'],
                        help='Sampling strategy to use for creating clips')
    parser.add_argument('--clip_length', type=int, default=16,
                        help='Number of frames per clip')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset')

    # Add more args as needed
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    return parser.parse_args()