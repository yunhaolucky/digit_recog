
from PyML import *
from PyML import ker
from PyML.classifiers import multi
from PyML.demo import demo2d

import csv
from PyML.datagen import sample

import matplotlib.pyplot as plt

def read_data(file_name):
   if file_name== 'train':
      data = vectorDatasets.VectorDataSet("train_new_train.data")
      data.attachLabels(Labels("train_new_label.data"))
      return data
   if file_name =='test':
      data = vectorDatasets.VectorDataSet("test.csv")
      return data

def multi_train(c,data,kernel_name):
   s=svm.SVM()
   #mc = multi.OneAgainstRest(svm.SVM(),C=1)
   mc = multi.OneAgainstOne(svm.SVM(),C=1)
   mc.cv(data,2)

def test_and_print(s,test):
   r=s.test(test)
   result_file = open("result.csv","wb")
   result_ob = csv.writer(result_file)
   result_ob.writerow(['ImageId','Label'])
   j=1
   for i in r.L:
      l=[j]
      l.append(i)
      result_ob.writerow(l)
      print l
      j=j+1

if __name__ == "__main__":
   data=read_data('train')
   data.normalize()
   test=read_data('test')
   s=svm.SVM()
   mc = multi.OneAgainstOne(svm.SVM(ker.Gaussian()),C=1,gamma=0.1)
   mc.train(data)
   test_and_print(mc,test)




