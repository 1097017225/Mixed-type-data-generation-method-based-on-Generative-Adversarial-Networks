
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

def probability_one_hot(data):
    return data.sum() / data.shape[0]


def probabilities_by_dimension_one_hot(data):
    return np.array([probability_one_hot(data[:, 15+j]) for j in range(63)])


def probability_count(data,i):
    pos_min = data >= i*0.1
    pos_max = data < (i+1)*0.1
    pos_rst = pos_min & pos_max
    return np.sum(pos_rst) / data.shape[0]

def probabilities_by_dimension_count(data):
    return [probability_count(data, j) for j in range(10)]


def main():

    #A = np.load("/root/pytorch/samples/pre-sample.npy", allow_pickle=True)
    #B = np.load("/root/pytorch/data/pre.matrix", allow_pickle=True)

    A = np.load("/root/pytorch/data/vae.matrix", allow_pickle=True)
    B = np.load("/root/pytorch/data/pre.matrix", allow_pickle=True)

    print(A.shape)
    print(B.shape)

    p_x_by_one_hot = probabilities_by_dimension_one_hot(A)
    p_y_by_one_hot = probabilities_by_dimension_one_hot(B)

    p_x_by_count = []
    p_y_by_count = []
    for i in range(15):
        p_x_by_count += probabilities_by_dimension_count(A[:, i])
        p_y_by_count += probabilities_by_dimension_count(B[:, i])

    p_x_by_count = np.array(p_x_by_count)
    p_y_by_count = np.array(p_y_by_count)

    #p_x = np.concatenate((p_x_by_count,p_x_by_one_hot))
    #p_y = np.concatenate((p_y_by_count,p_y_by_one_hot))

    #plt.figure(figsize=(10, 4), dpi=800)
    plt.scatter(p_x_by_one_hot, p_y_by_one_hot,c="b", marker="o")
    plt.scatter(p_x_by_count, p_y_by_count, c="k", marker="*")
    x1 = np.array([0,1])
    y2 = np.array([0,1])
    plt.plot(x1, y2, c="r", linewidth="0.5")
    plt.xlabel("$P_{fake}$",{'size':14})
    plt.ylabel("$P_{real}$",{'size':14})

    plt.scatter([0.02], [0.90], c="b", marker="o")
    plt.scatter([0.02], [0.82], c="k", marker="*")

    plt.text(0.05, 0.88, u"代表标签类型", size=15, alpha=1)
    plt.text(0.05, 0.80, u"代表数值类型", size=15, alpha=1)

    plt.plot([0,0.34], [0.94,0.94], c="k", linewidth="1")
    plt.plot([0.34, 0.34], [0.94, 0.78], c="k", linewidth="1")
    plt.plot([0.34, 0], [0.78, 0.78], c="k", linewidth="1")
    plt.plot([0, 0], [0.78, 0.94], c="k", linewidth="1")

    plt.savefig("VAE.jpg",dpi=720)
    plt.show()



if __name__ == "__main__":
    main()


