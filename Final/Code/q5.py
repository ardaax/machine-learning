import numpy as np
file = open("Movie Rate.txt", "r")
from sklearn import svm # Using Support vector regression

# Data parsing
data_list = list()
for line in file:
    splitted = line.split("\t")
    for val in range(len(splitted)):
        splitted[val] = splitted[val].strip()
        splitted[val] = float(splitted[val])
    data_list.append(splitted)

#   Split training and testinf data
train_data = list()
test_data = list()
for d in data_list:
    if d[0] != -1:
        train_data.append(d)
    else:
        test_data.append(d)

# Numpy arrays
train_data = np.array(train_data)
test_data = np.array(test_data)


Y_train = np.array(train_data.T[0])  # Actual Y values
train_data = np.delete(train_data, 0, 1)    # Delete the first column of train
test_data = np.delete(test_data, 0, 1)    # Delete the first column of test



regr = svm.SVR()
regr.fit(train_data, Y_train)
print(regr.predict(test_data))
