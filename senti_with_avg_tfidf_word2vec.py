# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('tweets@5k2.csv')
dataset.fillna(0, inplace=True)
# Cleaning the texts
import re
import nltk
#nltk.download('all')
#nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
from sklearn.metrics import pairwise_distances  
from sklearn.metrics.pairwise import cosine_similarity
corpus = []
for i in range(0, 7002):
    review = re.sub('@[\w\d]*','',dataset['Tweet'][i])
    if "#" in review:
        review = list(review)
        review.remove("#")                     
        review = "".join(review)
    review = re.sub('(?:https://|http://)[\w\d\.\/]*','',review).strip()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

for i in range(len(corpus)):
    if len(corpus[i])==0:
        print("hello")
        print(i)

with open('word2vec_model', 'rb') as handle:
    model = pickle.load(handle)



        
word2ven_matrix_of_titles=[]
for i in corpus:
    j=i.split()
    idf_avg = 0
    l=np.zeros(300,dtype="int64")
    for j1 in j:
        if j1 in model:
            j1_vec = model[j1]
            l=np.add(l,j1_vec)
    word2ven_matrix_of_titles.append(l/len(j))


import math
def find_tfidf_value(full,key):
    count=0
    for i in full:
        if key in i:
            count+=1
    return math.log(len(corpus)/count)


word2ven_matrix_of_titles1=[]
for i in corpus:
    j=i.split()
    idf_avg = 0
    l=np.zeros(300,dtype="int64")
    for j1 in j:
        if j1 in model:
            idf=find_tfidf_value(corpus,j1)
            tf=j.count(j1)/len(j)
            j1_vec = model[j1]*(idf*tf)
            l=np.add(l,j1_vec)
            idf_avg +=idf*tf
        else:
            idf=find_tfidf_value(corpus,j1)
            tf=j.count(j1)/len(j)
            idf_avg +=idf*tf
    if idf_avg==0:
        idf_avg=1
    word2ven_matrix_of_titles1.append(l/idf_avg)


word2ven_matrix_of_titles2= []
for i in range(len(word2ven_matrix_of_titles)):
    word2ven_matrix_of_titles2.append(word2ven_matrix_of_titles[i]*word2ven_matrix_of_titles1[i])


X = np.array(word2ven_matrix_of_titles)
y = (dataset.iloc[:, 1].values).astype(int)
y1 = (dataset.iloc[:,2].values).astype(int)
y2 = (dataset.iloc[:,3].values).astype(int)
y3 = (dataset.iloc[:,4].values).astype(int)
y4 = (dataset.iloc[:,5].values).astype(int)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.2, random_state = 0)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.2, random_state = 0)
X_train, X_test, y3_train, y3_test = train_test_split(X, y3, test_size = 0.2, random_state = 0)
X_train, X_test, y4_train, y4_test = train_test_split(X, y4, test_size = 0.2, random_state = 0)

predicted_values,y_pred,y1_pred,y2_pred,y3_pred,y4_pred = [],[],[],[],[],[]

for i in range(len(X_test)):
    idf_w2v_dist  = pairwise_distances(X_train, X_test[i].reshape(1, -1))
    indices = np.argsort(idf_w2v_dist.flatten())[0:1][0]
    y_pred.append(y_train[indices])
    y1_pred.append(y1_train[indices])
    y2_pred.append(y2_train[indices])
    y3_pred.append(y3_train[indices])
    y4_pred.append(y4_train[indices])
    predicted_values.append([y_train[indices],y1_train[indices],y2_train[indices],y3_train[indices],y4_train[indices]])

k=list(dataset["Tweet"][len(X_train):7002])
l5=[]
for i in range(len(predicted_values)):
    l5.append((k[i],predicted_values[i]))
dataFrame = pd.DataFrame(data=l5,columns=["tweet","centi"])



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm1 = confusion_matrix(y1_test, y1_pred)
cm2 = confusion_matrix(y2_test, y2_pred)
cm3 = confusion_matrix(y3_test, y3_pred)
cm4 = confusion_matrix(y4_test, y4_pred)

print("\nRESULT:\n")
print("confusion matrix are:")
print(cm)
print(cm1)
print(cm2)
print(cm3)
print(cm4)
p=((cm[0][0]+cm[1][1])/1400)
p1=((cm1[0][0]+cm1[1][1])/1400)
p2=((cm2[0][0]+cm2[1][1])/1400)
p3=((cm3[0][0]+cm3[1][1])/1400)
p4=((cm4[0][0]+cm4[1][1])/1400)
print("\n")
print("Positive = {:.2f} %".format(p*100))
print("Negative = {:.2f} %".format(p1*100))
print("Neutral = {:.2f} %".format(p2*100))
print("Question = {:.2f} %".format(p3*100))
print("Suggestion = {:.2f} %".format(p4*100))
total= (p1+p+p2+p3+p4)/5
print("Total = {:.2f} %".format(total*100))

