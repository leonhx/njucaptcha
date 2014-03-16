import numpy as np
import pylab as pl

train = np.load('data/clean_chars.npy')
train.shape = -1, 40, 40
n_samples = 40000
X = train[:n_samples]
X = (X[:, ::2, ::2] + X[:, 1::2, ::2] + X[:, ::2, 1::2] + X[:, 1::2, 1::2])/4
X.shape = len(X), -1

from sklearn.externals import joblib
cl = joblib.load('kmeans40k.pkl')

for l in np.unique(cl.labels_):
    labeled = X[cl.labels_ == l]
    for i in labeled:
        pl.figure(1)
        pl.show()
        pl.imshow(np.reshape(i, (20, 20)), cmap=pl.cm.Greys)
        if raw_input() == 'q':
            pl.close(1)
            break
        pl.close(1)
    print('continue?[y/N]'),
    n = raw_input()
    while not n or n not in 'NnYy':
        print('continue?[y/N]'),
        n = raw_input()
    if n[0].lower() == 'n':
        pl.close(1)
        break
