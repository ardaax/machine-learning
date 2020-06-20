from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import ann
"""TAKES APPROX 1 MIN TO RUN"""

iris = datasets.load_iris()


X = iris.data
Y = iris.target
order=range(np.shape(X)[0])
allocation=list(order)
np.random.shuffle(allocation)

Y_train = np.zeros(shape=(150,3))
for i in range(Y.shape[0]):
    if Y[i] == 0:
        Y_train[i] = [1, 0, 0]
    if Y[i] == 1:
        Y_train[i] = [0, 1, 0]
    if Y[i] == 2:
        Y_train[i] = [0, 0, 1]




train_X = np.zeros(shape=(100, 4))
train_Y = np.zeros(shape=(100, 3))
validation_X = np.zeros(shape=(25, 4))
validation_Y = np.zeros(shape=(25, 3))
test_X = np.zeros(shape=(25, 4))
test_Y = np.zeros(shape=(25, 3))

"""Split the data"""
for i in range(100):
    train_X[i] = X[allocation[i]]
    train_Y[i] = Y_train[allocation[i]]

cnt = 0
for i in range(100, 125):
    validation_X[cnt] = X[allocation[i]]
    validation_Y[cnt] = Y_train[allocation[i]]
    cnt += 1

cnt = 0
for i in range(125, 150):
    test_X[cnt] = X[allocation[i]]
    test_Y[cnt] = Y_train[allocation[i]]
    cnt += 1


""" Convert 25x3 matrix to 25x1 array with values 0,1,2"""
real_Y = list()
for i in range(len(test_Y)):
    ind = 0
    for index in range(len(test_Y[i])):
        if test_Y[i][index] == 1:
            ind = index
    real_Y.append(ind)


nnList = []
for i in range(10):
    nn = ann.NeuralNetwork(4, 4, 3, 0.3)
    nnList.append(nn)


valid_Y_converted = list()
cnt = 0
for i in range(100, 125):
    valid_Y_converted.append(Y[allocation[i]])
    cnt += 1

accuracy_list = list()
for nn in nnList:
    nn.fit(train_X, train_Y, 1000)
    predicted_Y = nn.predict(validation_X)

    pred_Y_converted = list()
    for i in range(len(predicted_Y)):    # For every sample
        max = predicted_Y[i][0]
        ind = 0
        for index in range(len(predicted_Y[i])):
                if predicted_Y[i][index] > max:
                    max = predicted_Y[i][index]
                    ind = index
        pred_Y_converted.append(ind)

    acc = 0  # accuracy
    for d in range(len(pred_Y_converted)):
        if pred_Y_converted[d] == valid_Y_converted[d]:
            acc += 1

    acc = (acc/len(pred_Y_converted))*100
    accuracy_list.append(acc)


""" Find the index of ann with max accuracy"""
min_ind = 0
max_acc = accuracy_list[0]
for i in range(len(accuracy_list)):
    if accuracy_list[i] > max_acc:
        max_acc = accuracy_list[i]
        min_ind = i

# print("This is the ann index %d and cost with lowest cost %f" % (min_ind, min_val))
test_pred_y = nnList[min_ind].predict(test_X)

# print("Actual values of testing samples", test_Y)
final_pred = list()
"""Find the max value index in every sample and assign that prediction to final pred list"""
for i in range(len(test_pred_y)):   # For all 25 elements
    max = test_pred_y[i][0]
    ind = 0
    for index in range(len(test_pred_y[i])):
        if test_pred_y[i][index] > max:
            max = test_pred_y[i][index]
            ind = index
    final_pred.append(ind)




print(final_pred)
print(real_Y)


"""Calculate accuracy"""

correct_num = 0  # Number of correctly classified
for i in range(len(final_pred)):
    if final_pred[i] == real_Y[i]:
        correct_num += 1

accuracy = (correct_num/len(final_pred))*100
print("The accuracy: ", accuracy)


"""     Confusion Matrix    """
data = {"y_Actual" : real_Y,
        "y_Predicted" : final_pred
        }
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()