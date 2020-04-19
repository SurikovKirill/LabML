from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

def colors(labels):
    return ['blue' if label == 'P' else 'red' for label in labels]


data = pd.read_csv("geyser.csv")
y = data['class'].values
X = data.drop(['class'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = AdaBoostClassifier(learning_rate=0.1, n_estimators=100)
model.fit(X_train, y_train)




plt.scatter(X[:, 0], X[:, 1], c = colors(y), cmap = plt.cm.Paired)
axes = plt.gca()
xLim = axes.get_xlim()
yLim = axes.get_ylim()
xs = np.linspace(xLim[0], xLim[1], 30)
ys = np.linspace(yLim[0], yLim[1], 30)
xCoords, yCoords = np.meshgrid(xs, ys)
xyCoords = np.vstack([xCoords.ravel(), yCoords.ravel()]).T
Z = model.staged_decision_function(xyCoords)
pred = model.staged_predict(X)

os.mkdir("data_geyser")

i = 1
for item, z in zip(pred, Z):
    plt.clf()

    plt.xlabel("Step number " + str(i))
    plt.scatter(X[:, 0], X[:, 1], c = colors(y), cmap = plt.cm.Paired)

    z = z.reshape(xCoords.shape)

    axes = plt.gca()

    axes.contour(xCoords, yCoords, z, colors = "black", levels = [-1, 0, 1], linestyles = ["--", "-", "--"])

    axes.scatter(X[:, 0], X[:, 1], facecolors = "none", edgecolors = colors(item), s = 100, linewidths = 1)
    plt.savefig("data_geyser" + "/" + "iteration" + str(i) + ".png")

    i += 1


# y_pred = model.staged_predict(X)
# for i, item in enumerate(y_pred, start=1):
#     plt.xlabel('Step number ' + str(i))
#     plt.scatter(X[:, 0], X[:, 1], c=colors(y), cmap=plt.cm.Paired)
#     ax = plt.gca()
#     ax.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors=colors(item), s=100, linewidths=1)
#     plt.savefig('scatters_chips/' + str(i) + '_iteration.png')











q_func = model.staged_score(X_test, y_test)
graphX = []
graphY = []
for i, item in enumerate(q_func, start=0):
    graphX.append(i)
    graphY.append(item)
plt.clf()
plt.plot(graphX, graphY, label="Accuracy depending on steps")
plt.xlabel("steps")
plt.ylabel("accuracy")
plt.legend()
plt.show()
