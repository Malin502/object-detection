#!/usr/bin/python
# -*- coding: sjis -*-

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target.astype(np.int64)
N = Y.size
Y2 = np.zeros(3 * N).reshape(N,3).astype(np.float32)

for i in range(N):
    Y2[i,Y[i]] = 1.0
    
index = np.arange(N)
np.save('train-x', X[index[index % 2 != 0]])
np.save('train-y', Y2[index[index % 2 != 0]] )
np.save('test-x', X[index[index % 2 == 0]])
np.save('test-y', Y[index[index % 2 == 0]])
np.save('train-y0', Y[index[index % 2 != 0]])

