import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
# pandas and numpy
# not so much of pandas but for read_csv which is more efficient than numpy.loadtxt
import numpy as np
import pandas as pd

# scikit-learn classifiers and cross validation utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

# scikit-learn dimension reduction
from sklearn.decomposition import PCA

# scikit-learn dataset processing utils
from sklearn.preprocessing import MinMaxScaler


# loading csv data into numpy array
def read_data(f, header=True, test=False):
    data = []
    labels = []

    csv_reader = csv.reader(open(f, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index = index + 1
        if header and index == 1:
            continue

        if not test:
            labels.append(int(row[0]))
            row = row[1:]

        data.append(np.array(np.int64(row)))
    return (data, labels)

PCA_COMPONENTS = 100


def doWork(train, labels, test):
    print "Converting training set to matrix"
    X_train = np.mat(train)

    print "Fitting PCA. Components: %d" % PCA_COMPONENTS
    pca = decomposition.PCA(n_components=PCA_COMPONENTS).fit(X_train)

    print "Reducing training to %d components" % PCA_COMPONENTS
    X_train_reduced = pca.transform(X_train)

    print "Fitting kNN with k=10, kd_tree"
    knn = KNeighborsClassifier(n_neighbors=10, algorithm="kd_tree")
    print knn.fit(X_train_reduced, labels)

    print "Reducing test to %d components" % PCA_COMPONENTS
    X_test_reduced = pca.transform(test)

    print "Preddicting numbers"
    predictions = knn.predict(X_test_reduced)

    print "Writing to file"
    write_to_file(predictions)

    return predictions

def my_kernel3(a,b):
    HA=a[np.newaxis,:,:].repeat(b.shape[0],axis=0)
    HB=b[:,np.newaxis,:].repeat(a.shape[0],axis=1)
    r=np.array([HA , HB]).min(axis=0).sum(axis=2)
    return r.astype(np.float32)

def my_kernel(x,u):
    n_samples , n_features = x.shape
    K = np.zeros(shape=(n_samples,1),dtype=np.float)
    for d in xrange(n_samples):
        K[d][0] = np.sum(np.minimum(x[d],u))
    return np.matrix(K)

def my_kernel2(x,y):
    x=x.tolist()
    y=y.tolist()
    r=[]
    for i in range(len(x)):
        x_temp=x[i]
        y_temp=y[i]
        result=0.0
        for j in range(len(x_temp)):
            if x_temp[j]>y_temp[j]:
                result+=y_temp[j]
            else:
                result+=x_temp[j]
        r.append(result)
    return np.matrix(r)

def doWorkTest():
    df = pd.read_csv('hog.csv')
    print "Finish Reading"
    df = df.astype('float64')
    ixs = np.arange(df.shape[0])
    splits = np.split(ixs, [14000, 28000])
    s=splits[0]
    min_max_scaler = MinMaxScaler()
    pca = PCA(n_components=80)
    X = df.ix[s,1:].copy()
    y = df.ix[s,0].copy()
    X = min_max_scaler.fit_transform(X)
    X = pca.fit_transform(X)
    svm = SVC(C=10,gamma=0.01,kernel='rbf').fit(X, y)
    df2 = pd.read_csv('hog_test.csv')
    df2 = df2.astype('float64')
    Xt = df2.copy()
    Xt = min_max_scaler.transform(Xt)
    return svm.predict(pca.transform(Xt))

def doWork2():
    df = pd.read_csv('hog.csv')
    df = df.astype('float64')
    print "Finish Reading"
    scalers = []
    pca_xfrms = []
    clfs = []
    ixs = np.arange(df.shape[0])
    splits = np.split(ixs, [14000, 28000])
    #tuned_parameters = [{'gamma': [1e-2], 'C': [10]}]
    tuned_parameters = [{'C': [10]}]
    for s in splits:
        min_max_scaler = MinMaxScaler()
        pca = PCA(n_components=80)

    # get training subset
        X = df.ix[s,1:].copy()
        y = df.ix[s,0].copy()
    
    # all the transformations
        X = min_max_scaler.fit_transform(X)
        X = pca.fit_transform(X)
    
    # train the classifier
        svm = GridSearchCV( SVC(kernel=my_kernel), tuned_parameters, cv=3, verbose=1 ).fit(X, y)
        #svm=SVC(kernel=my_kernel).fit(X,y)
    # store scaler, PCA transformer, and SVM classifier for this subset
        scalers.append(min_max_scaler)
        pca_xfrms.append(pca)
        clfs.append(svm)
    df2 = pd.read_csv('hog_test.csv')
    df2 = df2.astype('float64')
    preds = np.zeros((len(clfs), df2.shape[0]))

    i = 0
    for scaler, xfrm, clf in zip(scalers, pca_xfrms, clfs):
        Xt = df2.copy()
        Xt = scaler.transform(Xt)
        preds[i] = clf.predict(xfrm.transform(Xt))
    
        i += 1
    
    total_pred = [np.bincount(x).argmax() for x in preds.T.astype(int)]
    #min_max_scaler = MinMaxScaler()
    # pca = PCA(n_components=100)
    # print "Converting training set to matrix"
    # X_train = np.mat(train,dtype=float)
    # print "Fitting PCA. Components" 
    # #X = min_max_scaler.fit_transform(X_train)
    # X = pca.fit_transform(X)
    # svm=SVC(C=10, gamma=0.01,kernel='rbf')
    # svm.fit(X,labels)
    # p=svm.predict(pca.transform(test))
    return total_pred




def write_to_file(predictions):
    f = open("output.csv", "w")
    f.write('ImageId,Label')
    f.write('\n')
    for i in range(len(predictions)):
        f.write(str(i+1))
        f.write(',')
        f.write(str(int(predictions[i])))
        f.write("\n")
    f.close()


#if __name__ == '__main__':
    #p=doWork2()
    #train, labels = read_data("train.csv")
    #test, tmpl = read_data("test.csv", test=True)
    #print doWork(train, labels, test)