import os                          
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

PLOTS_DIR = "Data Visualization Plots"
os.makedirs(PLOTS_DIR, exist_ok=True)  


def save(filename):
    return os.path.join(PLOTS_DIR, filename)  


# Load CSV
df = pd.read_csv("chess_dataset.csv")

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["label"].value_counts())

# 1. Class balance bar chart
df["label"].value_counts().plot(kind="bar", color=["green", "orange", "red"])
plt.title("Class Distribution (Easy / Medium / Hard)")
plt.ylabel("Number of positions")
plt.tight_layout()
plt.savefig(save("class_balance.png"))    
plt.show()

feature_cols = [
    "material_diff",
    "mobility",
    "king_in_check",
    "pawn_structure",
    "attacked_pieces",
    "center_control",
    "king_safety",
    "game_phase",
    "passed_pawns",        # [NEW]
    "bishop_pair",         # [NEW]
    "open_file_rooks",     # [NEW]
    "weighted_mobility",   # [NEW]
]

# 3. Histograms per feature, grouped by label
for col in feature_cols:
    if col not in df.columns:
        print(f"  Skipping histogram for '{col}' — not in CSV yet.")
        continue
    plt.figure()
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        plt.hist(subset[col], bins=30, alpha=0.5, label=label)
    plt.title(f"Distribution of {col} by Difficulty")
    plt.legend()
    plt.savefig(save(f"hist_{col}.png"))  
    plt.close()

# 4. Correlation matrix
available_cols = [c for c in feature_cols if c in df.columns]  
corr = df[available_cols].corr()
plt.figure(figsize=(12, 10))                    
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(save("correlation_matrix.png"))     
plt.show()

# 5. Box plots for each feature across difficulty
for col in available_cols:
    plt.figure()
    sns.boxplot(x="label", y=col, data=df, order=["Easy", "Medium", "Hard"])
    plt.title(f"Box plot of {col} by Difficulty")
    plt.tight_layout()
    plt.savefig(save(f"boxplot_{col}.png"))    
    plt.close()

# 6. Feature importance via temporary Random Forest
X = df[available_cols]
y = df["label"]
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

plt.figure(figsize=(12, 6))                     
plt.bar(range(len(importances)), [importances[i] for i in indices])
plt.xticks(range(len(importances)), [available_cols[i] for i in indices], rotation=45)
plt.title("Feature Importance (Random Forest on full dataset)")
plt.tight_layout()
plt.savefig(save("feature_importance.png"))     
plt.show()

print("\nFeature importance scores:")
for i in indices:
    print(f"{available_cols[i]:<20}: {importances[i]:.4f}")
