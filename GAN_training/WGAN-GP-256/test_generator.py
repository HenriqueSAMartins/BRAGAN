import torch
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
from model import Generator
from utils import get_latest_checkpoint
from config import Z_DIM, CHANNELS_IMG, FEATURES_GEN, IMAGE_SIZE, device as DEVICE  # <-- import automático

# Hiperparâmetros (devem ser os mesmos do treinamento) -> OLHAR 'config.py'
# Z_DIM = 100  
# CHANNELS_IMG = 3
# FEATURES_GEN = 64
# IMAGE_SIZE = 64
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carregar automaticamente o último checkpoint
MODEL_PATH = get_latest_checkpoint("saved_models")
if MODEL_PATH is None:
    raise FileNotFoundError("Nenhum checkpoint encontrado na pasta 'saved_models'")
print(f"Carregando pesos de: {MODEL_PATH}")

# Criar pasta para resultados
os.makedirs("inference_output", exist_ok=True)

# Inicializar modelo e carregar pesos
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
gen.load_state_dict(checkpoint['gen_state_dict'])
gen.eval()

# Gerar imagens
with torch.no_grad():
    noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
    fake_images = gen(noise)

# Criar e salvar grid
img_grid = vutils.make_grid(fake_images, normalize=True)
output_path = "inference_output/generated_samples.png"
vutils.save_image(fake_images, output_path, normalize=True, nrow=8)
print(f"Imagens salvas em: {output_path}")

# Mostrar na tela
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Amostras Geradas pelo Generator")
plt.imshow(img_grid.permute(1, 2, 0).cpu())
plt.show()
