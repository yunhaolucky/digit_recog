import csv as csv
import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from skimage import data, color, exposure

def transfer_data(file_name):
	if file_name =='train':
		csv_file_object = csv.reader(open('train.csv', 'rb'))      
		header = csv_file_object.next()                             
		train=[]
		labels=[]                                                     
		for row in csv_file_object:
			labels.append(int(row[0]))
			train.append(map(int,row[1:]))
		train_file = open(file_name+"_new_train.data", "wb")
		train_ob = csv.writer(train_file)
		for i in train:
			train_ob.writerow((i))
		label_file = open(file_name+"_new_label.data", "wb")
		label_ob = csv.writer(label_file)
		for i in labels:
			label_ob.writerow([i])
	if file_name =='test':
		csv_file_object = csv.reader(open('test.csv', 'rb'))      
		header = csv_file_object.next()                             
		test=[]                                                    
		for row in csv_file_object:
			test.append(map(int,row))
		test_file = open(file_name+"_new.data", "wb")
		test_ob = csv.writer(test_file)
		for i in test:
			test_ob.writerow((i))

def transfer_to_hog(file_name):
	if file_name =='train':
		csv_file_object = csv.reader(open('train.csv', 'rb'))
		hog_write = csv.writer(open('hog.csv', 'wb'))      
		header = csv_file_object.next()
		imsize=(28,28)                                                                                  
		for row in csv_file_object:
			hogr=[]
			hogr.append(int(row[0]))
			image=map(int,row[1:])
			image=np.reshape(image, imsize)
			fd= hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1))
			hogr.extend(fd)
			fd= hog(image[2:26,2:26], orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1))
			hogr.extend(fd)
			hog_write.writerow(hogr)
	if file_name =='test':
		csv_file_object = csv.reader(open('test.csv', 'rb'))
		hog_write = csv.writer(open('hog_test.csv', 'wb'))      
		header = csv_file_object.next()
		imsize=(28,28)                                                                                  
		for row in csv_file_object:
			hogr=[]
			image=map(int,row)
			image=np.reshape(image, imsize)
			fd= hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1))
			hogr.extend(fd)
			fd= hog(image[2:26,2:26], orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1))
			hogr.extend(fd)
			hog_write.writerow(hogr)



if __name__ == "__main__":
	transfer_to_hog('test')
