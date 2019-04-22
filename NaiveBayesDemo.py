import numpy as np
from collections import Counter
from NaiveBayes import NaiveBayes


train_data = np.loadtxt('Data/mushroom_train.data', delimiter=',', dtype='<U1')

test_data = np.loadtxt('Data/mushroom_test.data', delimiter=',', dtype='<U1')
print(Counter(test_data[:, 0]))  # check if it is imbalanced data, the answer is no

_, d = test_data.shape

domain = list()
for i in range(1, d):
    domain.append(set(test_data[:, i]))

nb = NaiveBayes()
nb.train_with_uniform_dirichlet_prior(train_data[:, 0], train_data[:, 1:], domain, 0)

y = nb.predict(test_data[:, 1:])

print(np.sum(y == test_data[:, 0]) / len(y))

# domain = list()
# domain.append(set(test_data[:, 5]))
#
# nb = NaiveBayes()
# nb.train_with_uniform_dirichlet_prior(train_data[:, 0], train_data[:, [5]], domain, 10)
#
# y = nb.predict(test_data[:, [5]])
#
# print(np.sum(y == test_data[:, 0]) / len(y))

