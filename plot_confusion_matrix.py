"""
plot_confusion_matrix.py
========================
Generates and saves confusion matrix plots for both models.
Run: python plot_confusion_matrix.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from preprocess import load_data, preprocess
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Load data ──
df = load_data(os.path.join(ROOT, "data", "heart.csv"))
X_train, X_test, y_train, y_test, scaler = preprocess(df)

# ── Load models ──
models = {
    "RandomForest": joblib.load(os.path.join(ROOT, "models", "RandomForest.pkl")),
    "GradientBoosting": joblib.load(os.path.join(ROOT, "models", "GradientBoosting.pkl")),
}

bundle = joblib.load(os.path.join(ROOT, "models", "production_bundle.pkl"))
threshold = float(bundle.get("threshold", 0.5))

os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("CardioSense AI — Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)

for ax, (name, model) in zip(axes, models.items()):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Plot
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{name}\n(threshold={threshold})", fontsize=13, fontweight="bold")

    # Colorbar
    plt.colorbar(im, ax=ax)

    # Labels
    classes = ["No Disease (0)", "Disease (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Cell values
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center", fontsize=18, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("Actual Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    # Stats below
    sens = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0
    spec = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0
    acc  = round((tp + tn) / (tp + tn + fp + fn), 4)
    ax.set_xlabel(
        f"Predicted Label\n\nTP={tp}  TN={tn}  FP={fp}  FN={fn}\n"
        f"Sensitivity={sens}  Specificity={spec}  Accuracy={acc:.1%}",
        fontsize=10,
    )

plt.tight_layout()
out_path = os.path.join(ROOT, "models", "confusion_matrices.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"✅  Saved: {out_path}")
plt.show()
