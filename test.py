import torch
from torch.autograd import Variable
import numpy as np
import pickle
#import csv
# A = np.load("sample.matrix", allow_pickle=True)
#
# A = np.ceil(A).astype(np.int)
# file = open("3.csv",'a')
# file_csv = csv.writer(file)
# for i in A:
#     file_csv.writerow(i)
#e
# file.close()

A = np.load("samples/original-sample.matrix", allow_pickle=True)

# B = np.load("data/vae.matrix", allow_pickle=True)

# A[:,0]  = A[:,0]  * (max(B[:,0])  - min(B[:,0])) + min(B[:,0])
# A[:,1]  = A[:,1]  * (max(B[:,1])  - min(B[:,1])) + min(B[:,1])
# A[:,2]  = A[:,2]  * (max(B[:,2])  - min(B[:,2])) + min(B[:,2])
# A[:,3]  = A[:,3]  * (max(B[:,3])  - min(B[:,3])) + min(B[:,3])
# A[:,4]  = A[:,4]  * (max(B[:,4])  - min(B[:,4])) + min(B[:,4])
# A[:,5]  = A[:,5]  * (max(B[:,5])  - min(B[:,5])) + min(B[:,5])
# A[:,6]  = A[:,6]  * (max(B[:,6])  - min(B[:,6])) + min(B[:,6])
# A[:,7]  = A[:,7]  * (max(B[:,7])  - min(B[:,7])) + min(B[:,7])
# A[:,8]  = A[:,8]  * (max(B[:,8])  - min(B[:,8])) + min(B[:,8])
# A[:,9]  = A[:,9]  * (max(B[:,9])  - min(B[:,9])) + min(B[:,9])
# A[:,10] = A[:,10] * (max(B[:,10]) - min(B[:,10]))+ min(B[:,10])
# A[:,11] = A[:,11] * (max(B[:,11]) - min(B[:,11]))+ min(B[:,11])
# A[:,12] = A[:,12] * (max(B[:,12]) - min(B[:,12]))+ min(B[:,12])
# A[:,13] = A[:,13] * (max(B[:,13]) - min(B[:,13]))+ min(B[:,13])
# A[:,14] = A[:,14] * (max(B[:,14]) - min(B[:,14]))+ min(B[:,14])

# i = 15
# print(max(A[:,i]))
# print(max(B[:,i]))
#
# print(min(A[:,i]))
# print(min(B[:,i]))
print(A[0])
#
# indices = np.random.permutation(len(B))
# out_train = B[indices[0:9000]]
# out_test = B[indices[9000:10000]]
#
#pickle.dump(out_test, open("data/pre-vae-test" + '.matrix', 'wb'), -1)

