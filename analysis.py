import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("output/metrics.csv")

plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="epoch", y="loss", hue="eps", palette="viridis")
plt.yscale("log")
plt.title("PINN Loss Convergence across Epsilon")
plt.xlabel("Epoch")
plt.ylabel("Total Loss (Log Scale)")
plt.grid(True, which="both", ls="--")
plt.savefig("output/loss_convergence.png")
plt.show()