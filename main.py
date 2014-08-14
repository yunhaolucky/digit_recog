#from PyML import *
#from PyML import ker
import matplotlib.pyplot as plt
import csv as csv
import numpy as np
from sklearn import svm, metrics,cross_validation
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn import preprocessing

def read_data(file_name):
	if file_name =='train':
		csv_file_object = csv.reader(open('train.csv', 'rb'))      
		header = csv_file_object.next()                             
		train=[]
		labels=[]                                                     
		for row in csv_file_object:
			labels.append(int(row[0]))
			train.append(map(int,row[1:]))
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
		% (clf, metrics.classification_report(expected, predicted)))
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

def baseline(train, labels):
	classifier = svm.SVC(gamma=1)
	classifier.fit(train[:1000], labels[:1000])
	expected = labels[1000:2000]
	predicted =  classifier.predict(train[1000:2000])
	#print_report(classifier, expected, predicted)
	print("Classification report for classifier \n%s\n"
		% metrics.classification_report(expected, predicted))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
	return expected,predicted

def F_score(train,labels):
	sum_f=[[0,0,0,0,0,0,0,0,0,0] for i in range(len(train[0]))]
	for i in range(len(train)):
		label=labels[i]
		for j in range(len(train[0])):
			sum_f[j][label]+=train[i][j]

	feature=[0,0,0,0,0,0,0,0,0,0]
	for i in range(len(labels)):
		feature[labels[i]]+=1;

	total_sum=[sum(sum_f[i]) for i in range(len(sum_f))]
	total_mean=[(total_sum[i]+0.0)/len(train) for i in range(len(total_sum))]

	mean_f=[[0,0,0,0,0,0,0,0,0,0] for i in range(len(train[0]))]
	for i in range(len(sum_f)):
		for j in range(len(sum_f[0])):
			mean_f[i][j]=(sum_f[i][j]+0.0)/feature[j]

	f=[0 for i in range(len(train[0]))]
	numerator=[0 for i in range(len(train[0]))]
	divider=[0 for i in range(len(train[0]))]
	

	for i in range(len(f)):
	 	for j in range(10):
	 		numerator[i]+=(mean_f[i][j]-total_mean[i])*(mean_f[i][j]-total_mean[i])
	 		#print mean_f[i][j], total_mean[i]
	 	de_sum=[0.0 for i in range(10)]
	 	for m in range(len(train)):
	 		l=labels[m]
	 		de_sum[l]+=(train[m][i]+mean_f[i][l]+0.0)*(train[m][i]+mean_f[i][l]+0.0)
	 		#print train[m][i],mean_f[i][l]
	 	de_sum=[(de_sum[i]+0.0)/feature[i] for i in range(len(de_sum))]
	 	divider[i]=sum(de_sum)
	 	if divider[i] == 0:
	 		f[i]=100000
	 	else: 
	 		f[i]=numerator[i]/divider[i]
	return f

	#for i in range(len(sum_f)):

if remove_all_zero(data,total_sum):
	for i in range(len(total_sum)):
		if total_sum[len(total_sum)-1-i]==0:
			scipy.delete(second,len(total_sum)-1-i,1)




if __name__ == "__main__":
	labels,train=read_data('train')
	A=F_score(train,labels)
	#test=read_data('test')
	#print 'finish reading test'
	#clf = svm.LinearSVC()
	#baseline(train,labels)
	#cv_one_vs_all(clf,train[:500], labels[:500])
	#cv_one_vs_one(clf,train[:500], labels[:500])
	#expected,predicted=baseline(train,labels)
	#classifier = svm.LinearSVC()
	#scores = cross_validation.cross_val_score(classifier,train,labels, cv=5)
	#classifier.fit(train[:1000], labels[:1000])
	#expected = labels[500:1000]
	#predicted =  classifier.predict(train[1000:2000])
	#print_report(classifier, expected, predicted)
	#print("Classification report for classifier %s:\n%s\n"
	#	% (classifier, metrics.classification_report(expected, predicted)))
	#print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

