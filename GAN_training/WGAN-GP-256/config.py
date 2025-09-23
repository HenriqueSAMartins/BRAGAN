import torch

# =================== CONFIGURAÇÕES ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 
IMAGE_SIZE = 256
CHANNELS_IMG = 3
Z_DIM = 128 
NUM_EPOCHS = 801 
FEATURES_CRITIC = 128
FEATURES_GEN = 128
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

LOAD_MODEL = False

SAVE_MODEL = True
SAVE_EVERY = 100

# =================== PASTAS ===================
DATASET_PATH = "images_augmentation"
SAVE_MODEL_PATH = "saved_models"
GENERATED_IMG_PATH = "generated_images"
LOGS_PATH = "logs"
