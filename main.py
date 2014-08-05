#from PyML import *
#from PyML import ker
import matplotlib.pyplot as plt
import csv as csv
import numpy as np
from sklearn import svm, metrics,cross_validation
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier


def read_data(file_name):
	if file_name =='train':
		csv_file_object = csv.reader(open('train.csv', 'rb'))      
		header = csv_file_object.next()                             
		train=[]
		labels=[]                                                     
		for row in csv_file_object:
			labels.append(int(row[0]))
			train.append(row[1:])
		return labels,train
	if file_name == 'test':
		csv_file_object = csv.reader(open('test.csv', 'rb'))      
		header = csv_file_object.next()                             
		test=[]                                                    
		for row in csv_file_object:
			test.append(row)
		return test

    	#data = np.array(data)                       
def write_prediction(file_name,prediction):
	prediction_file = open(file_name+".csv", "wb")
	prediction_file_object = csv.writer(prediction_file)
	for i in prediction:
		prediction_file_object.writerow((i))

''' set all labels in target 0 and the rest 1'''
def trim_labels(target,labels):
	trimmed_labels=[]
	for i in labels:
		if i in target:
			trimmed_labels.append(0)
		else: 
			trimmed_labels.append(1)
	return trimmed_labels

'''test by hand'''
def manual_test(clf,train, train_labels,test,test_labels):
	clf.fit(train, train_labels)
	predicted = clf.predict(test)
	print_report(clf,test_labels,predicted)

'''print report and confusion matrx'''
def print_report(clf,expected,predicted):
	print("Classification report for classifier %s:\n%s\n"
		% (classifier, metrics.classification_report(test_labels, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


'''cross validation test with one vs all svm'''
def cv_one_vs_all(clf,train, labels):
	clf = OneVsRestClassifier(svm.LinearSVC())
	scores = cross_validation.cross_val_score(clf,train,labels, cv=5)
	print "one vs all", ("{0:.5f}".format(np.mean(scores)))
	return clf

'''cross validation test with one vs one svm'''
def cv_one_vs_one(clf,train, labels):
	#clf = OneVsOneClassifier(LinearSVC())
	scores = cross_validation.cross_val_score(clf,train,labels, cv=5)
	print "one vs all", ("{0:.5f}".format(np.mean(scores)))
	return clf

if __name__ == "__main__":
	labels,train=read_data('train')
	#test=read_data('test')
	print 'finish reading test'
	clf = svm.LinearSVC()
	cv_one_vs_all(clf,train[:500], labels[:500])
	cv_one_vs_one(clf,train[:500], labels[:500])


