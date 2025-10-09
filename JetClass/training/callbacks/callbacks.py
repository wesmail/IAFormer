# Generic imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.metrics import roc_curve, auc

# PyTorch Lightning imports
from lightning.pytorch.callbacks import Callback

# Framework imnports
from callbacks.cka import CKA


class TestCallback(Callback):
    def on_test_epoch_end(self, trainer, module):
        predictions = np.asarray(module.test_predictions)
        targets = np.asarray(module.test_targets)
        #attn_weights = np.asarray(module.attn_weights)
        #activations = np.asarray(module.activations)

        #print(f"shapes are {predictions.shape},  {targets.shape}, {attn_weights.shape} and {activations.shape}")

        # --------------------------------------------------------------------------------
        # ROC Curve
        # --------------------------------------------------------------------------------
        # create ROC curve
        fpr, tpr, threshold = roc_curve(targets, predictions)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            color="b",
            label="IAFormer (area = {:.3f}%)".format(auc(fpr, tpr) * 100),
        )
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive")
        plt.ylabel("True Positive")
        plt.title("ROC curve")
        plt.grid(linestyle="--", color="k", linewidth=0.9)
        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(trainer.log_dir + "_roc.png", dpi=300)

        # Save predictions, tpr
        np.save(f"predictions.npy", predictions)
        np.save(f"targets.npy", targets)
        np.save(f"tpr.npy", tpr)
        np.save(f"fpr.npy", fpr)
        
        # Save attention matrix and CKA matrix
        #np.save(f"attn_matrix.npy", attn_weights)
        #np.save(f"cka_matrix.npy", activations)
        
        # Calculate CKA Similarity 
        #cka = CKA()

        #cka_matrix = np.zeros((module.num_blocks, module.num_blocks))
        #for i in range(module.num_blocks):
        #    for j in range(module.num_blocks):
        #        X = activations[:, i]
        #        Y = activations[:, j]
        #        cka_matrix[i, j] = cka.linear_CKA(X, Y)

        #_ = plt.figure()
        #sns.heatmap(cka_matrix, annot=True, cmap="viridis")
        #plt.title("CKA Similarity Matrix")
        #plt.savefig(trainer.log_dir + "_cka.png", dpi=300)                 

        # free up the memory
        module.test_predictions.clear()
        module.test_targets.clear()
        #module.attn_weights.clear()
        #module.activations.clear()
