from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import re


def roc(counts, y):
    m = MultinomialNB(fit_prior=False)
    m.fit(counts, y)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y, m.predict_proba(counts)[:, 0], pos_label=0)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


folders = []
x = []
y = []
text = []
for i in os.walk('data_naive'):
    if len(i[2]) != 0:
        folders.append((i[0], i[2]))

for item in folders:
    for file in item[1]:
        if file.find('spmsg') != -1:
            y.append(1)
        if file.find('legit') != -1:
            y.append(0)
        file_handler = open('{0}/{1}'.format(item[0], file), 'r')
        for line in file_handler:
            text.extend(re.findall(r'\d+', line))
        x.append(' '.join(text))
        text.clear()

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(x)
roc(counts, y)

graphX, graphY = [], []
for j in range(1, 100, 1):
    model = MultinomialNB(class_prior=[j/100, 1-j/100])
    predictions = cross_val_score(model, counts, y, cv=10)
    graphX.append(j/100)
    graphY.append(predictions.mean())

plt.plot(graphX, graphY, label="Accuracy depending on prior probability")
plt.xlabel("prior probability")
plt.ylabel("accuracy")
plt.legend()
plt.show()
# j=10
# while (True):
#     j *= 10
#     model = MultinomialNB(alpha=0.001, class_prior=[0.1, 1/j])
#     predictions = cross_val_predict(model, counts, y, cv=10)
#     count = 0
#     for i in range(1090):
#         if y[i] == 0 and predictions[i] == 1:
#             count += 1
#             break
#     if count == 0:
#         print(1/j)
#         break
