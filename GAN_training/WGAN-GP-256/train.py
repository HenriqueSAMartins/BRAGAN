import os
import re
import csv
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
from model import Discriminator, Generator, initialize_weights
from utils import * 
from config import *


# =================== DATASET ===================

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                            if fname.lower().endswith((".png", ".jpg", ".jpeg"))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# =================== PASTAS UTILIZADAS ===================

os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
os.makedirs(GENERATED_IMG_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# =================== TRANSFORMAÇÕES E DADOS ===================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
])

dataset = CustomDataset(root_dir=DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =================== MODELOS ===================
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(disc)

gen_opt = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
disc_opt = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"{LOGS_PATH}/real")
writer_fake = SummaryWriter(f"{LOGS_PATH}/fake")

# =================== LOSS LOGGING ===================
loss_log_path = os.path.join(LOGS_PATH, "losses.csv")
with open(loss_log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "batch_idx", "loss_disc", "loss_gen"])


# =================== LOAD CHECKPOINT ===================
start_epoch = 0
if LOAD_MODEL:
    checkpoint_path = get_latest_checkpoint(SAVE_MODEL_PATH)
    if checkpoint_path is not None:
        print(f"Carregando checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
        disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("Nenhum checkpoint encontrado. Treinando do zero.")

# =================== TREINAMENTO ===================
start_time = time.time()
gen.train()
disc.train()
step = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    loop = tqdm(dataloader, leave=True)
    for batch_idx, real in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.size(0)

        # Treinar o Discriminador (ANTIGO)
        # for _ in range(CRITIC_ITERATIONS):
        #     noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
        #     fake = gen(noise)

        #     disc_real = disc(real).reshape(-1)
        #     disc_fake = disc(fake.detach()).reshape(-1)
        #     gp = gradient_penalty(disc, real, fake, device)
        #     loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA_GP * gp

        #     disc.zero_grad()
        #     loss_disc.backward()
        #     disc_opt.step()

        # # Treinar o Gerador
        # output = disc(fake).reshape(-1)
        # loss_gen = -torch.mean(output)

        # gen.zero_grad()
        # loss_gen.backward()
        # gen_opt.step()
        
        # Treinar o Discriminador
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake.detach()).reshape(-1)
            gp = gradient_penalty(disc, real, fake, device)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA_GP * gp

            disc.zero_grad()
            loss_disc.backward()
            disc_opt.step()

        # Treinar o Gerador
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
        fake = gen(noise)
        output = disc(fake).reshape(-1)
        loss_gen = -torch.mean(output)

        gen.zero_grad()
        loss_gen.backward()
        gen_opt.step()


        # Logs
        if batch_idx % 100 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_real.add_image("Real", img_grid_real, global_step=step)

            with open(loss_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, batch_idx, loss_disc.item(), loss_gen.item()])

            # Registrar perdas no TensorBoard
            writer_fake.add_scalar("Loss/Generator", loss_gen.item(), global_step=step)
            writer_real.add_scalar("Loss/Discriminator", loss_disc.item(), global_step=step)

            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss_disc=loss_disc.item(), loss_gen=loss_gen.item())
            step += 1

    # Salvar imagem gerada por época
    with torch.no_grad():
        fake = gen(fixed_noise)
        fake = fake * 0.5 + 0.5  # Desnormaliza
        torchvision.utils.save_image(fake, f"{GENERATED_IMG_PATH}/epoch_{epoch}.png", nrow=8)

    # Salvar checkpoint
    # torch.save({
    #     'gen_state_dict': gen.state_dict(),
    #     'disc_state_dict': disc.state_dict(),
    #     'gen_opt_state_dict': gen_opt.state_dict(),
    #     'disc_opt_state_dict': disc_opt.state_dict(),
    #     'epoch': epoch
    # }, os.path.join(SAVE_MODEL_PATH, f"checkpoint_epoch_{epoch}.pth"))
    
    # Salvar checkpoint condicional
    if SAVE_MODEL and (epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1):
        checkpoint = {
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'gen_opt_state_dict': gen_opt.state_dict(),
            'disc_opt_state_dict': disc_opt.state_dict(),
            'epoch': epoch
        }
        checkpoint_path = os.path.join(SAVE_MODEL_PATH, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(checkpoint, checkpoint_path)

# =================== SUMMARY ===================
end_time = time.time()
total_time = end_time - start_time
disk_usage = get_dir_size_mb(SAVE_MODEL_PATH)

with open(os.path.join(LOGS_PATH, "summary.txt"), "w") as f:
    f.write(f"Tempo total de treinamento: {total_time:.2f} segundos ({total_time/60:.2f} minutos)\n")
    f.write(f"Espaço ocupado em '{SAVE_MODEL_PATH}': {disk_usage:.2f} MB\n")