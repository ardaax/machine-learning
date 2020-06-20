from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
digits = datasets.load_digits()
k = 2

digits.values()

data = digits.data  #nd array of 1797,64
trg = digits.target  # Target values of 1797 data points.
trg_names = digits.target_names

m = data.shape[0]  # 1797 samples
feature_size = data.shape[1]  # 64 features
data_t = data.T  # Transposed

# Zero out the mean of the data
# For every feature calculate its mean and deduct
for d in range(feature_size):
    mean_j = (1/m)*np.sum(data_t[d])   #d'th feature's mean
    for i in range(m):  # Update
        data_t[d][i] = data_t[d][i] - mean_j    # Update the value

data = data_t.T  # Write changes to the original data


""" Covariance matrix """
cov_matrix = np.zeros((feature_size, feature_size))
for i in range(m):
    x_i = data[i].reshape(feature_size, 1)
    x_i_T = x_i.T
    cov_matrix += np.dot(x_i, x_i_T)
cov_matrix = cov_matrix/m


U, S, V = np.linalg.svd(cov_matrix)
W = U[:, :k]
x_reduced = np.dot(W.T, data.T)  # 2, 1797
x_T = x_reduced.T

# first_features = x_reduced[0],
# sec_features = x_reduced[1]


"""Keeping digits indices and dictionary of lists"""
dic = {}
for i in range(10):
    dic[str(i)] = list()

for i in range(len(trg)):
    dig = trg[i]
    str_dig = str(dig)
    dic[str_dig].append(i)


data_dic = {}   # This dictionary holds each digits data from x_reduced
for i in range(10):
    data_dic[str(i)] = np.zeros(shape=(len(dic[str(i)]), k))

# Fill data dictionary
index_list = [0] * 10  # Keeps the current index of the list
for i in range(m):
    dg = trg[i]  # Current digit
    data_dic[str(dg)][index_list[dg]] = x_T[i]
    index_list[dg] += 1

color_list = ["r", "b", "m", "g", "c"]

""" PART A FIRST FIVE DIGITS --- 0,1,2,3,4 --- 
0 -- Red, 1 - Blue, 2 - Magenta, 3 - Green, 4 - Cyan
"""
digit_list = ["0", "1", "2", "3", "4"]
for i in range(5):
    str_dig = digit_list[i]
    plt.scatter(data_dic[str_dig].T[0], data_dic[str_dig].T[1], c=color_list[i], label=digit_list[i])

plt.legend()
plt.title("Part A")
plt.show()


""" PART B LAST FIVE DIGITS --- 5, 6 ,7 ,8 ,9 --- 
5 -- Red, 6 - Blue, 7 - Magenta, 8 - Green, 9 - Cyan
"""
digit_list = ["5", "6", "7", "8", "9"]
for i in range(5):
    str_dig = digit_list[i]
    plt.scatter(data_dic[str_dig].T[0], data_dic[str_dig].T[1], c=color_list[i], label=digit_list[i])

plt.legend()
plt.title("Part B")
plt.show()


""" PART C EVEN DIGITS --- 0, 2, 4, 6, 8 --- 
0 -- Red, 2 - Blue, 4 - Magenta, 6 - Green, 8 - Cyan
"""
digit_list = ["0", "2", "4", "6", "8"]
for i in range(5):
    str_dig = digit_list[i]
    plt.scatter(data_dic[str_dig].T[0], data_dic[str_dig].T[1], c=color_list[i], label=digit_list[i])

plt.legend()
plt.title("Part C")
plt.show()


""" PART D ODD DIGITS --- 1, 3, 5, 7, 9 --- 
1 -- Red, 3 - Blue, 5 - Magenta, 7 - Green, 9 - Cyan
"""
digit_list = ["1", "3", "5", "7", "9"]
for i in range(5):
    str_dig = digit_list[i]
    plt.scatter(data_dic[str_dig].T[0], data_dic[str_dig].T[1], c=color_list[i], label=digit_list[i])

plt.legend()
plt.title("Part D")
plt.show()


""" PART E DIGITS --- 0, 3, 6, 9 --- 
0 -- Red, 3 - Blue, 6 - Magenta, 9 - Green
"""
digit_list = ["0", "3", "6", "9"]
for i in range(4):
    str_dig = digit_list[i]
    plt.scatter(data_dic[str_dig].T[0], data_dic[str_dig].T[1], c=color_list[i], label=digit_list[i])

plt.legend()
plt.title("Part E")
plt.show()