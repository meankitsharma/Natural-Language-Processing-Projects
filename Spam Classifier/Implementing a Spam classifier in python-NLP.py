
###################################  Implementing a Spam classifier  ##################################################

import pandas as pd

#Libraries ti clean the data
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# importing the Dataset
messages = pd.read_csv('C:/Users/meankitsharma93/Downloads/smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Cleaning the data
corpus=[]
Lemmatizer= WordNetLemmatizer()
stemmer= PorterStemmer() 

for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[Lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

#####################################  Method 1 - Using Bag of Words  ##################################################    
#Creating bag of words
from sklearn.feature_extraction.text import CountVectorizer  
cv=CountVectorizer(max_features=5000)  #It will choose only 5000 words from the all the words present in data
X=cv.fit_transform(corpus).toarray()

#Now creating dummy variable for dependent variable
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


#Performing train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0)

###################################  Now training model using Naive Bayes Classifier  ###################################
from sklearn.naive_bayes import MultinomialNB

spam_detect_NB=MultinomialNB().fit(X_train,y_train)

y_pred_NB=spam_detect_NB.predict(X_test)    

#Now checking the accuracy of our model

#Using confusion matrix
from sklearn.metrics import confusion_matrix

confusion_NB=confusion_matrix(y_test,y_pred_NB)

#Using accuracy score
from sklearn.metrics import accuracy_score

acc_score_NB=accuracy_score(y_test,y_pred_NB)
print(acc_score_NB)#0.9820

###################################  Now training model using SVM Classifier  ###################################
from sklearn import svm

clf = svm.SVC(kernel="linear")
spam_detect_svm=clf.fit(X_train,y_train)

y_pred_svm=spam_detect_svm.predict(X_test)    

#Now checking the accuracy of our model

#Using confusion matrix
from sklearn.metrics import confusion_matrix

confusion_svm=confusion_matrix(y_test,y_pred_svm)

#Using accuracy score
from sklearn.metrics import accuracy_score

acc_score_svm=accuracy_score(y_test,y_pred_svm)
print(acc_score_svm)#0.9865


#####################################  Method 2 - TF - IDF  ##############################################################    

#Creating the TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf=TfidfVectorizer(max_features=5000)
X=Tfidf.fit_transform(corpus).toarray()  

#Now creating dummy variable for dependent variable
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


#Performing train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0)

###################################  Now training model using Naive Bayes Classifier  ###################################
from sklearn.naive_bayes import MultinomialNB

spam_detect_NB=MultinomialNB().fit(X_train,y_train)

y_pred_NB=spam_detect_NB.predict(X_test)    

#Now checking the accuracy of our model

#Using confusion matrix
from sklearn.metrics import confusion_matrix

confusion_NB=confusion_matrix(y_test,y_pred_NB)

#Using accuracy score
from sklearn.metrics import accuracy_score

acc_score_NB=accuracy_score(y_test,y_pred_NB)
print(acc_score_NB)#0.9766

###################################  Now training model using SVM Classifier  ###################################
from sklearn import svm

clf = svm.SVC(kernel="linear")
spam_detect_svm=clf.fit(X_train,y_train)

y_pred_svm=spam_detect_svm.predict(X_test)    

#Now checking the accuracy of our model

#Using confusion matrix
from sklearn.metrics import confusion_matrix

confusion_svm=confusion_matrix(y_test,y_pred_svm)

#Using accuracy score
from sklearn.metrics import accuracy_score

acc_score_svm=accuracy_score(y_test,y_pred_svm)
print(acc_score_svm)#0.9838
    
###################################  Conclusion (Model Winner)  ###################################################

# SVM Classifier With Bag of Words Method with 98.65% accuracy


 
    
