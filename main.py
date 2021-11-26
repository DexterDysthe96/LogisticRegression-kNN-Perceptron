# Dexter Dysthe
# Dr. Krstovski
# B9122
# 22 November 2021

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model as lm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

plt.style.use('seaborn')

# ------------------------------------------ Q3 (a) ------------------------------------------ #

bank_loans = pd.read_csv('bankloans.csv')
x = bank_loans[['ed', 'employ', 'income', 'debtinc']].values
y = bank_loans['default'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=51796)


# ------------------------------------------ Q3 (b) ------------------------------------------ #

# ------------------------ Logistic Regression ------------------------ #

logistic = lm.LogisticRegression()
logistic.fit(x_train, y_train)

y_test_pred_lr = logistic.predict(x_test)
lr_accuracy_score = metrics.accuracy_score(y_test, y_test_pred_lr)

# ------------------------ k-Nearest Neighbors ------------------------ #

knn_accuracy_scores = []
k_vec = range(1, 51)
for k in k_vec:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    y_test_pred_knn = knn.predict(x_test)
    knn_accuracy_scores.append(metrics.accuracy_score(y_test, y_test_pred_knn))

plt.plot(np.array(k_vec), np.array(knn_accuracy_scores))
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy of k-NN')
plt.show()

# ---------------------------- Perceptron ----------------------------- #

x_train_norm = preprocessing.scale(x_train)
x_test_norm = preprocessing.scale(x_test)

perceptron = Perceptron(eta0=0.1)
perceptron.fit(x_train_norm, y_train)

y_test_pred_slp = perceptron.predict(x_test_norm)
slp_accuracy_score = metrics.accuracy_score(y_test, y_test_pred_slp)


# ------------------------------------------ Q3 (c) ------------------------------------------ #

accuracy_scores = pd.DataFrame({'Algorithm': ['Logistic Regression', '3-Nearest Neighbors', '5-Nearest Neighbors',
                                              '8-Nearest Neighbors', '9-Nearest Neighbors', '13-Nearest Neighbors',
                                              'Perceptron'],
                                'Accuracy Score on Test Set': [lr_accuracy_score, knn_accuracy_scores[0],
                                                               knn_accuracy_scores[1], knn_accuracy_scores[2],
                                                               knn_accuracy_scores[3], knn_accuracy_scores[4],
                                                               slp_accuracy_score]})

# As noted from the output, for the seed set at 51796, we have the following accuracy rankings in descending order:
# (1) 8-Nearest Neighbors
# (2) 9-Nearest Neighbors
# (3) Logistic Regression
# (3) 5-Nearest Neighbors
# (3) 13-Nearest Neighbors
# (6) 3-Nearest Neighbors
# (7) Perceptron
print(accuracy_scores)


