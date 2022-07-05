#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing


# In[2]:


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
    image = mpimg.imread('./train+validation/' + code)#read the image
    # print(image)
    train_images.append(image) #append the image
    train_labels.append(lab[0]) #lab[0] => without '\n'
train_file.close() #close train
# print(train_images[0])

for i in range(1, len(test_file_lines)):
    code = test_file_lines[i].strip('\n') #removes '\n'
    image = mpimg.imread('./test/' + code) #read the image
    #print(image)
    test_images.append(image) #append the image
test_file.close() #close test
# print(test_images[0])

for i in range(1, len(train_file_lines)):
    code, lab = train_file_lines[i].split(',') #split line by ','
    image = mpimg.imread('./train+validation/' + code) #read the image
    # print(image)
    validation_images.append(image) #append the image
    validation_labels.append(lab[0]) #lab[0] => without '\n'
validation_file.close() #close validation
# print(validation_images[0])


# In[3]:


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

#convert elements to int
test_images = test_images.astype(int)
train_images = train_images.astype(int)
train_labels = np.array(train_labels).astype(int)

# validation_labels = np.array(validation_labels).astype(int)

# validation_labels
# train_labels


# In[4]:


# lab 3 ai

# accuracy calculator function
#taken from the lab
def accuracy_score(true_labels, predicted_labels):
    return (true_labels == predicted_labels).mean()

#taken from the lab
# K-Nearest Neighbors - method
class Knn_classifier:

    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors = 3, metric = 'l2'): 

        if(metric == 'l2'):
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis = 1))
        elif(metric == 'l1'):
            distances = np.sum(abs(self.train_images - test_image), axis = 1)
        else:
            print('Error! Metric {} is not defined!'.format(metric))

        sort_index = np.argsort(distances) #returns the indexes that sort the array
        sort_index = sort_index[:num_neighbors]
        nearest_labels = self.train_labels[sort_index]
        histc = np.bincount(nearest_labels) #return the frequency of every value

        return np.argmax(histc)


    def classify_images(self, test_images, num_neighbors = 3, metric = 'l2'):
        num_test_images = test_images.shape[0] 
        predicted_labels = np.zeros((num_test_images))

        for i in range(num_test_images): 
            predicted_labels[i] = self.classify_image(test_images[i, :], num_neighbors = num_neighbors, metric = metric)

        return predicted_labels


# In[5]:


classifier = Knn_classifier(train_images, train_labels)


# In[6]:


# test_images = test_images.astype(int)
# type(test_images)

predicted_labels = classifier.classify_images(validation_images, 3, 'l2')


# In[7]:


# predicted_labels
# validation_labels


# In[8]:


# for i in range(0, len(validation_labels)):
#     validation_labels[i] = int(validation_labels[i])
# # print(validation_labels)

#convert elements to int
predicted_labels = predicted_labels.astype(int)

#convert list into np array of int
validation_labels = np.array(validation_labels).astype(int)


# In[9]:


# print(validation_labels, predicted_labels)

# accuracy calculator
accuracy_a = accuracy_score(validation_labels, predicted_labels)
print(accuracy_a)


# In[10]:


test_file = open("test.txt", 'r') #read file test
test_file_lines = test_file.readlines() #lines from test

write_file = open("knn_file.csv", 'w') #write into file
write_file.write("id,label\n") #append the first row
for i in range(1, len(test_file_lines)):
    # print(test_file_lines[i], len(test_file_lines[i]))
    if len(test_file_lines[i]) == 20:
        write_file.write(f"{test_file_lines[i][:-1]},{predicted_labels[i]}\n") #append to file without '\n'
    else:
        write_file.write(f"{test_file_lines[i]},{predicted_labels[i]}\n") #append to file no '\n'

write_file.close() #close write


# In[ ]:




