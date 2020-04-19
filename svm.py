from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

regularization_strength = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
gamma = ['scale', 'auto', 0.2, 0.25, 0.75]


data = pd.read_csv("chips.csv")
# data['class'] = pd.factorize(data['class'])[0]
y = data['class'].values
X = data.drop(['class'], axis=1).values
# X_norm = normalize(X)
# for strength in regularization_strength:
#     for kern in kernel:
#         if kern == 'poly' or kern == 'rbf' or kern == 'sigmoid':
#             for g in gamma:
#                 svc = SVC(C=strength, kernel=kern, gamma=g)
#                 scores = cross_val_score(svc, X, y, scoring='f1')
#                 print("Accuracy: %0.2f (+/- %0.2f) with params: %s and %s with gamma = %s" % (
#                 scores.mean(), scores.std() * 2, strength, kern, g))
#         else:
#             svc = SVC(C=strength, kernel=kern)
#             scores = cross_val_score(svc, X, y, scoring='f1')
#             print("Accuracy: %0.2f (+/- %0.2f) with params: %s and %s" % (scores.mean(), scores.std() * 2, strength, kern))




def colors (labels):
    return ['blue' if label == 'P' else 'red' for label in labels]

clf = SVC(kernel='sigmoid', C=100.0, gamma='auto')
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=colors(y), cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()