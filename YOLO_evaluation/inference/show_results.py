import pandas as pd
import matplotlib.pyplot as plt

# Carregar o CSV
df = pd.read_csv("runs/detect/train3/results.csv")

# Gráfico 1: Evolução da mAP
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("Evolução do mAP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico 2: Perdas de treinamento
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["train/box_loss"], label="Box Loss")
plt.plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
plt.plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Perda de Treinamento por Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico 3: Precision e Recall
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precisão e Revocação")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
