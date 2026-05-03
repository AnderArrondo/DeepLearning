import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import numpy as np

#AI Generated
def show_conf(cm, class_names):
    os.makedirs("./2assign/results", exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.colorbar()

    # labels en ejes
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.title("Confusion Matrix")

    # números dentro de cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center", color="white")

    plt.savefig("./2assign/results/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()