import torch
import torch.utils.data
import torchvision.utils as vutils
from models import Generator
import argparse
import os
import yaml



def generate_image(generator, label, noise_dim, device):
    # Create random noise
    noise = torch.randn(1, noise_dim, 1, 1, device=device)
    label = torch.tensor([label], device=device)  # Assuming label is a single integer (e.g., 0, 1, 2 for 3 classes)

    # Generate image
    with torch.no_grad():  # No need to compute gradients
        generated_image = generator(noise, label, alpha=1.0)  # Modify alpha as needed
        return generated_image


def main():
    # Arguments
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to trained generator weights')
    parser.add_argument('--label', type=int, required=True, help='Label for image generation (e.g., 0, 1, or 2)')
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimension of the noise vector (default: 100)')
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Directory to save the generated image')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Device to run the inference on')
    parser.add_argument('--resolution', type=str, default='64', choices=['64', '128','256'],
                        help='Device to run the inference on')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    generator = torch.load(args.weights, map_location=device)
    generator = generator.to(device)
    generator.eval()

    generated_image = generate_image(generator, args.label, args.noise_dim, device)

    generated_image = (generated_image + 1) / 2

    output_path = os.path.join(args.output_dir, f"generated_label_{args.label}.png")
    vutils.save_image(generated_image.detach(), output_path, normalize=True)

    print(f"Generated image saved at: {output_path}")


if __name__ == "__main__":
    main()