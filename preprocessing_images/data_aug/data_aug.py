# HS = Horizontal Shift
# VS = Vertical Shift
# HF = Horizontal Flip
# RT = Rotation

# BG = Brightness
# CT = Contrast
# ST = Saturation

# Libraries
import cv2
import random
import os
import numpy as np

# Função para preencher a imagem após transformações
def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

# Função para aplicar deslocamento horizontal (HS)
def horizontal_shift(img, nome, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w * ratio
    if ratio > 0:
        img = img[:, :int(w - to_shift), :]
    if ratio < 0:
        img = img[:, int(-1 * to_shift):, :]
    img = fill(img, h, w)
    cv2.imwrite(f"./images_augmentation/HS-{nome}", img)

# Função para aplicar deslocamento vertical (VS)
def vertical_shift(img, nome, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h * ratio
    if ratio > 0:
        img = img[:int(h - to_shift), :, :]
    if ratio < 0:
        img = img[int(-1 * to_shift):, :, :]
    img = fill(img, h, w)
    cv2.imwrite(f"./images_augmentation/VS-{nome}", img)

# Função para ajustar o brilho (BG)
def brightness(img, low, high, nome):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f"./images_augmentation/BG-{nome}", img)

# Função para aplicar flip horizontal (HF)
def horizontal_flip(img, flag, nome):
    if flag:
        img = cv2.flip(img, 1)
        cv2.imwrite(f"./images_augmentation/HF-{nome}", img)

# Função para aplicar rotação (RT)
def rotation(img, angle, nome):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite(f"./images_augmentation/RT-{nome}", img)


# Nova função: Contraste (CT)
def contrast(img, nome, alpha_range=(0.5, 1.5)):
    alpha = random.uniform(alpha_range[0], alpha_range[1])
    img_contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    cv2.imwrite(f"./images_augmentation/CT-{nome}", img_contrast)

# Nova função: Saturação (ST)
def saturation(img, nome, factor_range=(0.5, 1.5)):
    factor = random.uniform(factor_range[0], factor_range[1])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    img_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f"./images_augmentation/ST-{nome}", img_sat)

# Cria a pasta "images_augmentation" se não existir
if not os.path.exists("./images_augmentation"):
    os.makedirs("./images_augmentation")

# Lendo as imagens da pasta "images"
pasta = './images_cropped_256_blur'
for diretorio, subpastas, arquivos in os.walk(pasta):
    if len(arquivos) == 0:
        print("Não há imagens na pasta 'images'.")
    else:
        for arquivo in arquivos:
            if arquivo.endswith(".jpg") or arquivo.endswith(".png"):
                img = cv2.imread(os.path.join(pasta, arquivo))
                if img is not None:
                    # Aplica todas as técnicas de data augmentation
                    # Transformações originais
                    horizontal_shift(img, arquivo, ratio=0.3)
                    vertical_shift(img, arquivo, ratio=0.3)
                    horizontal_flip(img, flag=True, nome=arquivo)
                    rotation(img, angle=5, nome=arquivo)
                    
                    # Novas transformações
                    brightness(img, low=0.9, high=1.4, nome=arquivo)
                    contrast(img, nome=arquivo, alpha_range=(0.9, 1.3))
                    saturation(img, nome=arquivo, factor_range=(0.9, 1.3))
                else:
                    print(f"Erro ao carregar a imagem: {arquivo}")