import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# scikit-learn classifiers and cross validation utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

# scikit-learn dimension reduction
from sklearn.decomposition import PCA

# scikit-learn dataset processing utils
from sklearn.preprocessing import MinMaxScaler

row=3
col=3
df = pd.read_csv('./train.csv')
#df = df.astype('float64')
f,  l= plt.subplots(ncols=col,nrows=row)
#l=(ax1, ax2,ax3,ax4)

imsize = (28, 28)

for i in range(row):
	for j in range(col):
		l[i][j].matshow(np.reshape(df.ix[random.randint(0,df.shape[0]),1:], imsize), cmap='gray_r')
		l[i][j].axis('off')

f.tight_layout();

#plt.imshow(f)
plt.show()

