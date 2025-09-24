import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./original_dataset/data/01295.csv")

features = [col for col in df.columns]

n_features = len(features)
fig, axes = plt.subplots(int(n_features/3), 3, figsize=(12, 2.5 * n_features), sharex=True)
axes = axes.flatten()

for i, col in enumerate(features):
    axes[i].plot(df.index, df[col], label=col)
    axes[i].set_ylabel(col)
    axes[i].legend(loc="upper right")
    axes[i].grid(True)

axes[-1].set_xlabel("row index")
plt.suptitle("feature-wise plots", fontsize=14)
plt.tight_layout(rect=[0,0,1,0.97])
plt.show()
plt.savefig("./data")
