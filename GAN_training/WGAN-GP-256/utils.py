import os
import re
import torch

def gradient_penalty(critic, real, fake, device="cpu"):
    """
    Calcula a penalidade do gradiente para WGAN-GP.
    real: imagens reais
    fake: imagens geradas
    critic: modelo discriminador
    device: cuda ou cpu
    """
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1), device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    # Score do critic
    mixed_scores = critic(interpolated)

    # Gradiente dos scores em relação às imagens interpoladas
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

# =================== FUNÇÕES AUXILIARES ===================
def save_checkpoint(state, filename):
    torch.save(state, filename)

def get_latest_checkpoint(folder="saved_models"):
    checkpoints = [f for f in os.listdir(folder) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(re.findall(r"epoch_(\d+)", x)[0]), reverse=True)
    return os.path.join(folder, checkpoints[0])

def get_dir_size_mb(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)