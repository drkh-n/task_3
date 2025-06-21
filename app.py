import streamlit as st
import torch
import matplotlib.pyplot as plt

# Load your trained generator model
@st.cache_resource  # cache model load so it loads once
def load_model():
    model = torch.load("generator.pth", map_location="cpu")
    model.eval()
    return model

model = load_model()

# App title
st.title("MNIST Digit Generator")
st.write("Enter a digit (0-9) and generate 5 sample images!")

# Input: digit
digit = st.number_input("Enter digit (0-9)", min_value=0, max_value=9, step=1)

# Generate button
if st.button("Generate"):
    latent_dim = 100  # or whatever your model uses
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))

    for ax in axes:
        z = torch.randn(1, latent_dim)
        label = torch.tensor([digit])

        # Assuming model takes (z, label)
        with torch.no_grad():
            generated = model(z, label)  # adapt if model API is different
        
        img = generated.squeeze().numpy()

        ax.imshow(img, cmap='gray')
        ax.axis('off')

    st.pyplot(fig)
