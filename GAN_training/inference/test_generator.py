import torch
import torchvision.utils as vutils
import os
from model import Generator
from utils import get_latest_checkpoint
from config import Z_DIM, CHANNELS_IMG, FEATURES_GEN, IMAGE_SIZE, device as DEVICE

# ======= Configurações =======
N_IMAGES = 200  # Número de imagens a serem geradas
SAVE_DIR = "fake"
LOG_FILE = os.path.join(SAVE_DIR, "log.txt")
BATCH_SIZE = 128  # Número de imagens por batch

# ======= Setup do modelo =======
MODEL_PATH = get_latest_checkpoint("saved_models")
if MODEL_PATH is None:
    raise FileNotFoundError("Nenhum checkpoint encontrado na pasta 'saved_models'")
print(f"Carregando pesos de: {MODEL_PATH}")

os.makedirs(SAVE_DIR, exist_ok=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
gen.load_state_dict(checkpoint['gen_state_dict'])
gen.eval()

# ======= Descobrir o índice inicial =======
existing_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
existing_indices = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
start_idx = max(existing_indices) + 1 if existing_indices else 0

# ======= Abrir log =======
log = open(LOG_FILE, "a")

# ======= Geração =======
with torch.no_grad():
    remaining = N_IMAGES
    current_idx = start_idx

    while remaining > 0:
        batch_size = min(BATCH_SIZE, remaining)
        noise = torch.randn(batch_size, Z_DIM, 1, 1).to(DEVICE)
        fake_images = gen(noise)

        for i in range(batch_size):
            img_name = f"{current_idx}.png"
            save_path = os.path.join(SAVE_DIR, img_name)

            vutils.save_image(fake_images[i], save_path, normalize=True)
            noise_vector = noise[i].cpu().numpy().flatten()

            log_line = f"{img_name} -> noise: {noise_vector.tolist()}\n"
            log.write(log_line)

            print(log_line.strip())

            current_idx += 1

        remaining -= batch_size

log.close()
print(f"\n{N_IMAGES} imagens salvas na pasta '{SAVE_DIR}' e log registrado em 'log.txt'")
