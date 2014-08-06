
from PyML import *
from PyML import ker
from PyML.demo import demo2d


from PyML.datagen import sample

import matplotlib.pyplot as plt

def read_data(file_name):
   data = vectorDatasets.load_libsvm_format(file_name)
   return data



def train_wok(c,data,cv_num=5):
   s=svm.SVM(C=c)
   return s.cv(data,cv_num)

def train(c,data,cv_num,kernel_name):
   s=svm.SVM(getkernel(kernel_name),C=c)
   return s.cv(data,cv_num)


def plot(x,Y,labels,x_lable,y_lable,name):
   plt.figure()
   for i in range(len(Y)):
      plt.plot(x,Y[i],'-',label=labels[i])
   plt.xlabel(x_lable)
   plt.ylabel(y_lable)
   plt.legend(loc=4)
   plt.xscale('log')
   #plt.xticks(x, x)
   plt.savefig('start_latex/'+name+'.jpg')

def getkernel(kernel_name):
   if kernel_name[0]=='g':
      return ker.Gaussian(gamma = float(kernel_name[1:]))
   if kernel_name[0]=='p':
      return ker.Polynomial(degree = int(kernel_name[1:]))

def create_list(start,end, multi):
   x=[]
   i=start
   x.append(i)
   while i<end:
      i*=multi
      x.append(i)
   return x

def plot_single_c(C, data, name):
    acc=[]
    bacc=[]
    auc=[]
    for c in C:
      result=train_wok(x,data,5)
      acc.append(result.getSuccessRate())
      bacc.append(result.getBalancedSuccessRate())
      auc.append(result.getROC())
    Y=[]
    Y.append(acc)
    Y.append(bacc)
    Y.append(auc)
    labels=['acc','balanced acc','auc']
    plot(X,Y,labels,'C','',name)



def plot_c(start, end, multi,data,kernel_list,name):
   x=[]
   i=start
   while i<=end:
      i *= multi
      x.append(i)
   a=[];
   labels=[];
   for kernel_name in kernel_list:
      labels.append(kernel_name)
      k_acc=[]
      for i in x:
         r=train_wok(i,data,5)
         k_acc.append(r.getSuccessRate())
      a.append(k_acc)
   plot(x,a,labels,'C','acc',name)
   plt.clf()
   return x,a


def plot_g(start, end, multi,data,c_list,name):
   x=[]
   i=start
   a=[]
   while i<=end:
      i *= multi
      x.append(i)
   labels=[];
   for c in c_list:
      labels.append('C='+str(c))
      k_acc=[]
      for i in x:
         r=train(c,data,2,'g'+str(i))
         k_acc.append(r.getBalancedSuccessRate())
      a.append(k_acc)
   plot(x,a,labels,'gamma','acc',name)
   plt.clf()
   return x,a

def plot_p(start, end, multi,data,c_list,name):
   x=range(start,end,multi)
   labels=[]
   a=[]
   for c in c_list:
      labels.append('C='+str(c))
      k_acc=[]
      for i in x:
         r=train(c,data,2,'p'+str(i))
         k_acc.append(r.getBalancedSuccessRate())
      a.append(k_acc)
   plot(x,a,labels,'gamma','acc',name)
   plt.clf()
   return x,a

if __name__ == "__main__":
   #data=read_data('motif')
   #data.normalize()
   #x,a=plot_c(0.0001,1000,10,data,['g0.01','g1','g10','p1','p3'],'c')
   #x,a=plot_c(0.0001,1000,1.5,data,['p1'],'c2')
   #r=train(1,data,5,'p1')
   #plot_single_c(create_list(0.00001,100000,2),data,'c3')
   #x,a=plot_g(0.0001,1000,10,data,[0.01,1,10],'g')
   #x,a=plot_g(0.0001,1000,10,data,[0.01],'g1')

   #x,a=plot_p(1,15,3,data,[0.01,1,10],'p')
   #x,a=plot_p(1,15,3,data,[0.01],'p1')
   #demo2d.getData()
   #data=demo2d.data
   #x,a=plot_c(0.0001,1000,10,data,['g0.01','g1','g10','p1','p3'],'c')
   #x,a=plot_c(0.0001,1000,10,data,['p1'],'c1')
   #temp.decisionSurface(c)
   #demo2d.decisionSurface()
   #demo2d.decisionSurface(svm.SVM(C=0.1))
   #demo2d.decisionSurface(svm.SVM(C=10))
   #demo2d.decisionSurface(svm.SVM(C=1000))
   
   #demo2d.decisionSurface(svm.SVM(ker.Gaussian(gamma=0.1)))
   #demo2d.decisionSurface(svm.SVM(ker.Gaussian(gamma=1)))
   #demo2d.decisionSurface(svm.SVM(ker.Gaussian(gamma=10)))
   #demo2d.decisionSurface(svm.SVM(ker.Gaussian(gamma=100)))




