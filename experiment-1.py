# FOR BI-LEVEL CONFUSION MATRIX
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

y_true_bin = np.array([1,0,1,1,0,1,0,0,1,0])
y_pred_bin = np.array([1,0,1,0,0,1,1,0,1,0])

cm_bin = confusion_matrix(y_true_bin, y_pred_bin)

plt.figure()
plt.imshow(cm_bin)
plt.title("Bi-level Confusion Matrix")
plt.colorbar()
for i in range(cm_bin.shape[0]):
    for j in range(cm_bin.shape[1]):
        plt.text(j, i, cm_bin[i, j], ha='center')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# FOR MULTI-LEVEL CONFUSION MATRIX
y_true_multi = np.array([0,1,2,1,0,2,1,0,2,1])
y_pred_multi = np.array([0,2,2,1,0,1,1,0,2,0])

cm_multi = confusion_matrix(y_true_multi, y_pred_multi)

plt.figure()
plt.imshow(cm_multi)
plt.title("Multi-class Confusion Matrix")
plt.colorbar()

for i in range(cm_multi.shape[0]):
    for j in range(cm_multi.shape[1]):
        plt.text(j, i, cm_multi[i, j], ha='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Multi-Level Accuracy:", accuracy_score(y_true_multi, y_pred_multi))
