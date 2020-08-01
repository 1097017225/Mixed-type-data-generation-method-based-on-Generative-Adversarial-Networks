import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False

fx = open("error_x.txt", "r")
fy = open("error_y.txt", "r")
fvae = open("error_vae_y.txt", "r")
A = []
B = []
C = []
for line1, line2, line3 in zip(fx, fy, fvae):
    A.append(float(line1))
    B.append(float(line2))
    C.append(float(line3))
#print(A)
#print(B)
fx.close()
fy.close()
fvae.close()

A = np.array(A)
B = np.array(B)
C = np.array(C)

plt.figure(1)
plt.scatter(B[0:15], A[0:15], c="k", marker="*")
plt.scatter(B[15:], A[15:], c="b", marker="o")
x1 = np.array([0, 1])
y2 = np.array([0, 1])
plt.plot(x1, y2, c="r", linewidth="0.5")
plt.xlabel("$E_{fake}$",{'size':14})
plt.ylabel("$E_{real}$",{'size':14})


plt.scatter([0.02], [0.90], c="b", marker="o")
plt.scatter([0.02], [0.82], c="k", marker="*")

plt.text(0.05, 0.88, u"代表标签类型", size=15, alpha=1)
plt.text(0.05, 0.80, u"代表数值类型", size=15, alpha=1)

plt.plot([0,0.34], [0.94,0.94], c="k", linewidth="1")
plt.plot([0.34, 0.34], [0.94, 0.78], c="k", linewidth="1")
plt.plot([0.34, 0], [0.78, 0.78], c="k", linewidth="1")
plt.plot([0, 0], [0.78, 0.94], c="k", linewidth="1")
plt.savefig("GAN.jpg",dpi=720)




plt.figure(2)
plt.scatter(C[0:15], A[0:15], c="k", marker="*")
plt.scatter(C[15:], A[15:], c="b", marker="o")
x1 = np.array([0, 1])
y2 = np.array([0, 1])
plt.plot(x1, y2, c="r", linewidth="0.5")
plt.xlabel("$E_{fake}$",{'size':14})
plt.ylabel("$E_{real}$",{'size':14})
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
