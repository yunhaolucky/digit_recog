import csv as csv

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

if __name__ == "__main__":
	transfer_data('train')
