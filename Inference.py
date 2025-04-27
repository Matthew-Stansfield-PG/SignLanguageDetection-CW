import streamlit as st
import torch
import torch.utils.data
import torchvision.utils as vutils
from models import Generator
import os
import yaml
from PIL import Image


def generate_image(generator, label, noise_dim, device, alpha=1.0):
    # Create random noise
    noise = torch.randn(1, noise_dim, 1, 1, device=device)
    label = torch.tensor([label], device=device)  # Assuming label is a single integer (e.g., 0, 1, 2 for 3 classes)

    # Generate image
    with torch.no_grad():  # No need to compute gradients
        generated_image = generator(noise, label, alpha=alpha)  # Modify alpha as needed
        return generated_image


def main():
    # Streamlit widgets for input
    st.title("GAN Image Generator")
    st.write("Generate images from a pre-trained GAN model")

    # Upload the pre-trained model weights file
    uploaded_file = st.file_uploader("Upload model weights (.pt file)", type="pt")
    if uploaded_file is not None:
        # Load the weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config from a file or define it manually
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Initialize the generator model with the config
        generator = Generator(config['nz'], config['ngf'], config['nc'], 3, resolution=64).to(device)
        checkpoint = torch.load(uploaded_file, map_location=device)
        generator.load_state_dict(checkpoint)
        generator.eval()

        # Inputs for label and noise_dim
        label = st.slider("Choose a label for image generation", 0, 1, 2)  # Assuming 3 classes (0, 1, 2)
        noise_dim = st.number_input("Noise dimension", min_value=1, value=100, step=1)
        resolution = st.selectbox("Select resolution", ["64", "128", "256"])

        # Button to generate the image
        if st.button("Generate Image"):
            # Adjust the resolution dynamically (if required)
            generator = Generator(config['nz'], config['ngf'], config['nc'], 3, resolution=int(resolution)).to(device)
            generator.load_state_dict(checkpoint, strict=False)
            generator.eval()

            # Generate image
            generated_image = generate_image(generator, label, noise_dim, device)

            # Normalize the generated image (to [0, 1] if needed)
            generated_image = (generated_image + 1) / 2  # Normalize if the model output is in the range [-1, 1]

            # Convert to numpy and display
            generated_image = generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            generated_image = (generated_image * 255).astype('uint8')

            # Convert to Image for displaying in Streamlit
            img = Image.fromarray(generated_image)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img,
                         caption=f"Generated Image (Label: {label}, Noise Dim: {noise_dim}, Resolution: {resolution})",
                         width=128)

            # Option to download the generated image
            img_path = "generated_image.png"
            img.save(img_path)
            st.download_button(label="Download Image", data=open(img_path, "rb").read(), file_name=img_path,
                               mime="image/png")


if __name__ == "__main__":
    main()
