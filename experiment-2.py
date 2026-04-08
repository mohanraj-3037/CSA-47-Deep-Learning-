import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

w = load_wine()
d = pd.DataFrame(w.data, columns=w.feature_names)
d['t'] = w.target
d = d[d['t'] != 2]

X, y = d.drop('t', axis=1), d['t']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)

m = DecisionTreeClassifier(random_state=1).fit(Xtr, ytr)
yp = m.predict(Xte)

print(accuracy_score(yte, yp))
print(classification_report(yte, yp, target_names=w.target_names[:2]))
print(precision_score(yte, yp), recall_score(yte, yp), f1_score(yte, yp))

sns.heatmap(confusion_matrix(yte, yp), annot=True, fmt='d', cmap='PuBuGn',
            xticklabels=w.target_names[:2], yticklabels=w.target_names[:2])
plt.show()