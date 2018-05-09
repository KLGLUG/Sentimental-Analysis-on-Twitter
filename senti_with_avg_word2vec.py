'''
;;; Multi_class Sentiment Analysis on Twitter using Machine learning model
;;; Copyright © 2018 Uday Kiran Kondreddy <udaykiran.kondreddy@gmail.com>
;;; Copyright © 2018 Farhaan Ahmed Shaik <farhaanfsk@gmail.com>
;;; Copyright © 2018 Subhani Kurra <subhanikurra4@gmail.com>
;;; Copyright © 2018 Naga Teja Mamidapaka <nagatejam@gmail.com>
;;;
;;; This file is part of Sentimental Analysis on Twitter.
;;;
;;; This is free Code; you can redistribute it and/or modify it
;;; under the terms of the GNU General Public License as published by
;;; the Free Software Foundation; either version 3 of the License, or (at
;;; your option) any later version.
;;;
;;; This code is distributed in the hope that it will be useful, but
;;; WITHOUT ANY WARRANTY; without even the implied warranty of
;;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;;; GNU General Public License for more details.
;;;
;;; See <http://www.gnu.org/licenses/>.'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import WordNetLemmatizer

# Importing the dataset
dataset = pd.read_csv('tweets@5k.csv')
#dataset.fillna(0,inplace=True)
# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.metrics import pairwise_distances  
from sklearn.metrics.pairwise import cosine_similarity
corpus = []
d1,d2,d3,d4,d5={},{},{},{},{}
for i in range(0, 5000):
    review= dataset['Tweet'][i]
    #review = re.sub('@[\w\d]*','',dataset['Tweet'][i])
    if "#" in review:
        review = list(review)
        review.remove("#")
        review.remove("@")                     
        review = "".join(review)
    review = re.sub('(?:https://|http://)[\w\d\.\/]*','',review).strip()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    for each_word in review:
        if dataset['positive'][i]==1:
            d1.setdefault(each_word,0)
            d1[each_word]+=1
        elif dataset['negative'][i]==1:
            d2.setdefault(each_word,0)
            d2[each_word]+=1
        elif dataset['neutral'][i]==1:
            d3.setdefault(each_word,0)
            d3[each_word]+=1
        elif dataset['question'][i]==1:
            d4.setdefault(each_word,0)
            d4[each_word]+=1
        elif dataset['suggestion'][i]==1:
            d5.setdefault(each_word,0)
            d5[each_word]+=1
    review = ' '.join(review)
    corpus.append(review)


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

k=list(dataset["Tweet"][len(X_train):5000])
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
p=((cm[0][0]+cm[1][1])/1000)
p1=((cm1[0][0]+cm1[1][1])/1000)
p2=((cm2[0][0]+cm2[1][1])/1000)
p3=((cm3[0][0]+cm3[1][1])/1000)
p4=((cm4[0][0]+cm4[1][1])/1000)
print("\n")
print("Positive = {:.2f} %".format(p*100))
print("Negative = {:.2f} %".format(p1*100))
print("Neutral = {:.2f} %".format(p2*100))
print("Question = {:.2f} %".format(p3*100))
print("Suggestion = {:.2f} %".format(p4*100))
total= (p1+p+p2+p3+p4)/5
print("Total = {:.2f} %".format(total*100))

#0.7108974358974358
