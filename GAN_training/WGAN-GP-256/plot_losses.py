import csv
import matplotlib.pyplot as plt
import pandas as pd

# Carregar o CSV
df = pd.read_csv("logs/losses.csv")

# Aplicar média móvel
window = 10
df["loss_gen_smooth"] = df["loss_gen"].rolling(window).mean()
df["loss_disc_smooth"] = df["loss_disc"].rolling(window).mean()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df["loss_gen"], alpha=0.3, label="Generator Loss (raw)")
plt.plot(df["loss_disc"], alpha=0.3, label="Discriminator Loss (raw)")
plt.plot(df["loss_gen_smooth"], label="Generator Loss (smoothed)", linewidth=2)
plt.plot(df["loss_disc_smooth"], label="Discriminator Loss (smoothed)", linewidth=2)

plt.xlabel("Batch Updates")
plt.ylabel("Loss")
plt.title("WGAN-GP Training Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/loss_plot.png")
plt.show()
