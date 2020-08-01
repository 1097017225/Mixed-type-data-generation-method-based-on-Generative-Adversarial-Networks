import numpy as np
import matplotlib.pyplot as plt

def probability_one_hot(data):
    return data.sum() / data.shape[0]


def probabilities_by_dimension_one_hot(data):
    return np.array([probability_one_hot(data[:, 15+j]) for j in range(63)])


def probability_count(data,i,j):
    pos_min = data >= i*0.1*(Max_count[j]-Min_count[j]) + Min_count[j]
    pos_max = data < (i+1)*0.1*(Max_count[j]-Min_count[j]) + Min_count[j]
    pos_rst = pos_min & pos_max
    return np.sum(pos_rst) / data.shape[0]

def probabilities_by_dimension_count(data,i):
    return [probability_count(data, j,i) for j in range(10)]


def main():

    # A = np.load("/root/pytorch/samples/pre-sample.npy", allow_pickle=True)
    # B = np.load("/root/pytorch/data/pre.matrix", allow_pickle=True)

    A = np.load("/root/pytorch/samples/original-sample.matrix", allow_pickle=True)
    B = np.load("/root/pytorch/data/original.matrix", allow_pickle=True)


    p_x_by_one_hot = probabilities_by_dimension_one_hot(A)
    p_y_by_one_hot = probabilities_by_dimension_one_hot(B)

    global Max_count
    global Min_count
    Max_count = [max(B[:, 0]),max(B[:, 1]),max(B[:, 2]),max(B[:, 3]),max(B[:, 4]),max(B[:, 5]),max(B[:, 6]),max(B[:, 7]),max(B[:, 8]),max(B[:, 9]),max(B[:, 10]),max(B[:, 11]),max(B[:, 12]),max(B[:, 13]),max(B[:, 14])]
    Min_count = [min(B[:, 0]), min(B[:, 1]), min(B[:, 2]), min(B[:, 3]), min(B[:, 4]), min(B[:, 5]), min(B[:, 6]),
                 min(B[:, 7]), min(B[:, 8]), min(B[:, 9]), min(B[:, 10]), min(B[:, 11]), min(B[:, 12]), min(B[:, 13]),
                 min(B[:, 14])]

    p_x_by_count = []
    p_y_by_count = []
    for i in range(15):
        p_x_by_count += probabilities_by_dimension_count(A[:, i],i)
        p_y_by_count += probabilities_by_dimension_count(B[:, i],i)

    p_x_by_count = np.array(p_x_by_count)
    p_x_by_count = np.array(p_x_by_count)

    p_x = np.concatenate((p_x_by_count,p_x_by_one_hot))
    p_y = np.concatenate((p_y_by_count,p_y_by_one_hot))


    plt.scatter(p_x, p_y)
    x1 = np.array([0,1])
    y2 = np.array([0,1])
    plt.plot(x1,y2)
    plt.xlabel("$P_{fake}$",{'size':14})
    plt.ylabel("$P_{real}$",{'size':14})
    plt.show()


if __name__ == "__main__":
    main()


