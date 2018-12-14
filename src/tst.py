'''
test.py - scratchwork sandbox
12 Dec 2018 - Johnston, Germer, and Stucki
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 32)
y = np.arange(0, 32)
arr = np.zeros((y.size, x.size))

cx = 7.
cy = 16.
r = 2.7

# The two lines below could be merged, but I stored the mask
# for code clarity.
#mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2

def neighbors(mat, rad, r, c, dropNan=True):
    ''' Grabs indexes of neighbors within some radius of (r,c).  Note that rad must be greater than 1 since we are dealing with indexes '''
    tmpR = np.arange(0,mat.shape[0])
    tmpC = np.arange(0,mat.shape[1])
    mask = np.power(tmpR[:,np.newaxis]-r,2)+np.power(tmpC[np.newaxis,:]-c,2) < np.power(rad,2)
    return np.where(mask==True)

for i1,i2 in zip(np.arange(6,0,-.3),np.arange(0,6,.3)):
    idxs = neighbors(arr,i1,cy,cx)
    arr[idxs] = 100*i2

# This plot shows that only within the circle the value is set to 123.
plt.figure()
plt.pcolormesh(x, y, arr)
plt.colorbar()
plt.show()