#############################################################################
#This program is a simple email spam detector
#It utilises Naive Bayes algorithm to construct a MultinomialNB classifier
#############################################################################

#import libraries
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
import pickle as p

#set view options for dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

#dataset consists of 1499 spam(label 1) emails and 3672 ham(label 0) emails
#dataset can be downloaded from:
#https://www.kaggle.com/venky73/spam-mails-dataset
#load training file as pandas dataframe
df = pd.read_csv('/Users/username/Documents/spam_ham_dataset.csv')

#function to save classifier model and vectorizer for reuse
def save(model, file_name):
    with open(file_name, 'wb') as model_file:
        p.dump(model, model_file)
    print ("file saved")

#split training set(word and label) into train and test, set test size to 20 percent
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'],random_state=1, test_size=0.2)
#variable to keep apostrophes in words
word_pattern = "(?u)\\b[\\w']+\\b"
#initiase CountVectorizer() to change text from dataset into numerical form of vectors
vect = CountVectorizer(token_pattern=word_pattern)
#transform into train and test
X_train_vectorized = vect.fit_transform(X_train)
X_test_vectorized = vect.transform(X_test)
#initiase naive bayes algorithm
naive_b = MultinomialNB()
#train model
naive_b.fit(X_train_vectorized, y_train)
#declare predictions variable to test model
predictions = naive_b.predict(X_test_vectorized)

#print accuracy score, confusion matrix and classification report to evaluate the model
print("Accuracy score: " , accuracy_score(y_test, predictions))
print("CONFUSION_MATRIX: \n", confusion_matrix(y_test, predictions))
print("CLASSIFICATION REPORT: \n",classification_report(y_test, predictions))

#save trained model and vecorizer for reuse(if required)
save(naive_b, "clf_model.mdl")
save(vect, "vectorizer.pickle")

#################################################################
#test the trained classifier model on new data
#################################################################

#text from the training file
#ham
a="Subject: enron methanol ; meter # : 988291  this is a follow up to the note i gave you on monday , 4 / 3 / 00 { preliminary  flow data provided by daren } .  please override pop ' s daily volume { presently zero } to reflect daily  activity you can obtain from gas control .  this change is needed asap for economics purposes ."
#spam
b="Subject: photoshop , windows , office . cheap . main trending  abasements darer prudently fortuitous undergone  lighthearted charm orinoco taster  railroad affluent pornographic cuvier  irvin parkhouse blameworthy chlorophyll  robed diagrammatic fogarty clears bayda  inconveniencing managing represented smartness hashish  academies shareholders unload badness  danielson pure caffein  spaniard chargeable levin "
#spam
c="Subject: report 01405 !  wffur attion brom est inst siupied 1 pgst our riwe asently rest .  tont to presyou tew cons of benco 4 . yee : fater 45 y . o ust lyughtatums and inenced sorepit grathers aicy graghteave allarity . oarity wow to yur coons , as were then 60 ve mers of oite .  ithat yoit ? ! berst thar ! enth excives 2004 . . ."
#spam
d="Subject: vic . odin n ^ ow  berne hotbox carnal bride cutworm dyadic  guardia continuous born gremlin akin counterflow hereafter vocabularian pessimum yaounde cannel bitch penetrate demagogue arbitrary egregious adenosine rubin gil luminosity delicti yarmulke sauterne selfadjoint agleam exeter picofarad consulate dichotomous boyhood balfour spheric frey pillory hoosier fibonacci cat handful  "
#ham
e="Subject: tenaska iv july  darren :  please remove the price on the tenaska iv sale , deal 384258 , for july and enter the demand fee . the amount should be $ 3 , 902 , 687 . 50 .  thanks ,  megan"

#randomly generated text
#spam
f="Hi mr Smith, Click here to enter! Win instant prize now! "
#ham
g="Hi tom, please find a report attached below"
#ham
h="My name is Catherine, I am the new lead of the marketing department. Kind Regards"
#spam
i="Bitcoin cash pize waiting for you $$$9999!"
#spam
j="Insurance compensation for your car accident awaits!!!s"

#create a list
emails=[a,b,c,d,e,f,g,h,i,j]
#vectorize new data
new_data_vectorized = vect.transform(emails)
#declare predication variable and feed vectorized data through the classifier
predictions_label = naive_b.predict(new_data_vectorized)
#print the list of predicted labels for each list item
#label 0 idicates ham, label 1 indicates spam
print(predictions_label)

