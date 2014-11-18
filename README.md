Digit Recognition with SVM
===========

### Data --  [MNIST](http://yann.lecun.com/exdb/mnist/)
  * The training data set has 784 column(label + images) and 42000 rows.
  * The digit images contain grey levels(0 - 255) and is centered in a 28x28 image by computing the center of mass of the pixels.
  *  The first 40 digit images are shown as follows:
  !()

#### Baseline 
  * run SVM with linear kernel on the training data 
  * Using accuracy to evalute: 0.908



The best result I get is 98.45% by using orientation histogram features and rbf kernels.

- Baseline
  I use SVM with linear kernel. 
  - One vs One 0.908
  - One vs Rest 0.930

- rbf kernel
 	The best performance is achieved by hyperparameter set[ C=10, gamma = 0.7 ] error rate with (PCA = 80) = 3%

- SVM with Binary Decision Tree
	Build a binary decision tree based on the euclidean distance between each labels' mean. error rate = 13%.

- Feature Extraction:
  
  I try 2 methods to reduce the size of features:1)F-scores. 2)PCA:Principal component analysis
  - F-score
  	Use F-scores(variance between groups/variance within groups) to pick the most N relavant features. I test with N = 50, 100, 200. error rate > 10%.
  - PCA
  	Use sklean.decomposition.PCA to reduce the dimensionality of training data. I choose n_components = 200 as the result is good and running time is relatively short. error rate = 3.54%.


* Support Vector Machine Note
  * [Kernel function] (http://yunhaocsblog.wordpress.com/2014/07/23/kernel-function/)
  * [SMO] (http://yunhaocsblog.wordpress.com/2014/07/22/smo-algorithm/)
  * [THE EFFECTS OF HYPERPARAMETERS IN SVM] (http://yunhaocsblog.wordpress.com/2014/07/27/the-effects-of-hyperparameters-in-svm/)
  * [ROC CURVES](http://yunhaocsblog.wordpress.com/2014/07/20/roc-curves/)

  I also try to using Orientation Histogram as features. I use scimage.hog to get HOG features and predict with rbf and intersection kernel.
  - rbf (C=10 gamma =0.01 PCA_component = 200) error rate = 1.55%.
  - intersection (PCA_component = 200) error rate = 2.8%
