import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generator architecture
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.main(z)
        return img.view(-1, 1, 28, 28)

latent_dim = 100  # adjust if you used different latent dim

# Load model
@st.cache_resource
def load_model():
    model = Generator(latent_dim)
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("MNIST Digit Generator")
st.write("Click the button to generate 5 random samples!")

if st.button("Generate"):
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for ax in axes:
        z = torch.randn(1, latent_dim)
        with torch.no_grad():
            img = model(z)
        img = img.squeeze().numpy()
        ax.imshow((img + 1) / 2, cmap='gray')  # scale from [-1,1] to [0,1]
        ax.axis('off')
    st.pyplot(fig)
