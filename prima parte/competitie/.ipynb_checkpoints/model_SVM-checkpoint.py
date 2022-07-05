#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.image as mpimg
from sklearn import svm
from sklearn import preprocessing


# In[14]:


train_images = []
train_labels = []
test_images = []
validation_images = []
validation_labels = []

train_file = open("train.txt", 'r') #read file train
test_file = open("test.txt", 'r') #read file test
validation_file = open("validation.txt", 'r') #read file validation
train_file_lines = train_file.readlines() #lines from train
test_file_lines = test_file.readlines() #lines from test


for i in range(1, len(train_file_lines)):
    code, lab = train_file_lines[i].split(',') #split line by ','
    image = mpimg.imread('./train+validation/' + code) #read the image
    # print(image)
    train_images.append(image) #append the image
    train_labels.append(lab[0]) #lab[0] => without '\n'
train_file.close() #close train
# print(train_images[0])

for i in range(1, len(test_file_lines)):
    code = test_file_lines[i].strip('\n') #removes '\n'
    image = mpimg.imread('./test/' + code)#read the image
    #print(image)
    test_images.append(image) #append the image
test_file.close() #close test
# print(test_images[0])

for i in range(1, len(train_file_lines)):
    code, lab = train_file_lines[i].split(',')#split line by ','
    image = mpimg.imread('./train+validation/' + code)#read the image
    # print(image)
    validation_images.append(image) #append the image
    validation_labels.append(lab[0]) #lab[0] => without '\n'
validation_file.close() #close validation
# print(validation_images[0])


# In[15]:


#transform lists into np arrays
train_images = np.array(train_images)
test_images = np.array(test_images)
validation_images = np.array(validation_images)

# print(train_images.flatten().shape)
# print(test_images.flatten().shape)
# print(validation_images.flatten().shape)

# arrays reshape
# flatten => return a copy of the array collapsed into one dimension
# reshape => reshapes an array without changing the data of the array
train_images = train_images.flatten().reshape(8000, 768) #6144000 / 768, 16 * 16 * 3
test_images = test_images.flatten().reshape(2819, 768) #2164992 / 768, 16 * 16 * 3 
validation_images = validation_images.flatten().reshape(8000, 768) #6144000 / 768, 16 * 16 * 3


# In[16]:


# lab 4 ai

# accuracy calculator function
#taken from the lab
def acc_calculator(true_labels, predicted_labels):
    return (true_labels == predicted_labels).mean()

#taken from the lab
def normalize(train_images, test_images, type = None):
    scaler = preprocessing.Normalizer(norm = 'l2')
    scaler.fit(train_images)
    scaler_train = scaler.transform(train_images)
    scaler_test = scaler.transform(test_images)
    return scaler_train, scaler_test


# In[17]:


#normalize in norm l2
scaled_train, scaled_test = normalize(train_images, validation_images, type = 'l2')

# print(scaled_train, scaled_test, sep = '\n\n\n\n')


# In[18]:


svm_clsf = svm.SVC(C = 0.9, kernel = 'poly')

# print(svm_clsf)

svm_clsf.fit(scaled_train, train_labels) #train svm_clsf

predict = svm_clsf.predict(scaled_test) #prediction based on scaled_test

accuracy = acc_calculator(validation_labels, predict) #calculate the accuracy

print(accuracy)


# In[22]:


test_file = open("test.txt", 'r') #read file test
test_file_lines = test_file.readlines() #lines from test

write_file = open("svm_file.csv", 'w') #write into file
write_file.write("id,label\n") #append the first row
for i in range(1, len(test_file_lines)):
    # print(test_file_lines[i], len(test_file_lines[i]))
    if len(test_file_lines[i]) == 20:
        write_file.write(f"{test_file_lines[i][:-1]},{predict[i]}\n") #append to file without '\n'
    else:
        write_file.write(f"{test_file_lines[i]},{predict[i]}\n") #append to file no '\n'

write_file.close() #close write


# In[ ]:




