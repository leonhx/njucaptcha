import numpy as np

from sklearn.cluster import KMeans

train = np.load('data/clean_chars.npy')
train.shape = -1, 40, 40
n_samples = 40000
X = train[:n_samples]
X = (X[:, ::2, ::2] + X[:, 1::2, ::2] + X[:, ::2, 1::2] + X[:, 1::2, 1::2])/4
X.shape = len(X), -1

cl = KMeans(n_clusters=45)
cl.fit(X)

from sklearn.externals import joblib
joblib.dump(cl, 'kmeans40k.pkl')
