import numpy as np
import matplotlib.pyplot as plt

def probability(data,i):
    pos_min = data > i*0.1
    pos_max = data < (i+1)*0.1
    pos_rst = pos_min & pos_max
    return np.sum(pos_rst) / data.shape[0]

def probabilities_by_dimension(data):
    return [probability(data, j) for j in range(10)]


B = np.load("/root/pytorch/data/count-test.matrix", allow_pickle=True)
A = np.load("/root/pytorch/samples/sample-test.npy", allow_pickle=True)

test1 = []
test2 = []
for i in range(15):
    test1 += probabilities_by_dimension(A[:, i])
    test2 += probabilities_by_dimension(B[:, i])

test1 = np.array(test1)
test2 = np.array(test2)
print(test1.shape)
# print(np.round(test1,4))
# print(np.round(test2,4))

plt.scatter(test1, test2)
x1 = np.array([0, 1])
y2 = np.array([0, 1])

plt.plot(x1, y2)
plt.xlabel("Psample")
plt.ylabel("Ptest")
plt.show()
