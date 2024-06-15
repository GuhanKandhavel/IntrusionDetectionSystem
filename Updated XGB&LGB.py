#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
import itertools
from xgboost import XGBClassifier
from tabulate import tabulate


# In[2]:


train=pd.read_csv('UNSW_NB15_Testing.csv')


# In[3]:


train.head()


# In[4]:


train.dtypes


# In[5]:


train.describe(include='object')


# In[6]:


train["attack_cat"].unique()


# In[7]:


train["attack_cat"].value_counts()


# In[8]:


sns.countplot(x=train['attack_cat'])


# In[9]:


train["proto"].value_counts()


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


label_encoder=LabelEncoder()


# In[12]:


train["attack_cat"]=label_encoder.fit_transform(train["attack_cat"])


# In[13]:


train["proto"]=label_encoder.fit_transform(train["proto"])


# In[14]:


train["service"]=label_encoder.fit_transform(train["service"])


# In[15]:


train["state"]=label_encoder.fit_transform(train["state"])


# In[16]:


print(label_encoder)


# In[21]:


train.head()


# In[22]:


X_train = train.drop(['proto'], axis=1)
Y_train = train['proto']


# In[23]:


import matplotlib.pyplot as plt
plt.plot(X_train, Y_train)
plt.show()


# In[24]:


rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X_train, Y_train)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
selected_features = [v for i, v in feature_map if i==True]

selected_features


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import warnings
from xgboost import XGBClassifier
import time
from sklearn.metrics import f1_score


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.50, random_state=42)
warnings.filterwarnings("ignore")
# Create and train LightGBM model
model = lgb.LGBMClassifier()
model.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(x_test)
# Evaluate model performance
accuracy1 = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
#f1=f1_score(y_pred, y_test)
print("==> Accuracy of LightGBM : %.2f%% <==" % (accuracy1 * 100.0))
#print("==> Time:"," %s seconds <==" % (time.time() - start_time))
#print("==>F1_score=",f1,"<==")

models = []
models.append(('XGB', XGBClassifier(eta=0.1, gamma=5)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    start_time = time.time()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    predictions = [value for value in y_pred]
    #f=f1_score(y_pred, y_test)
    # evaluate predictions
    accuracy2 = accuracy_score(y_test, predictions)
    print("==> Accuracy of XGB : %.2f%% <==" % (accuracy2 * 100.0))
   # print("==> Time:"," %s seconds <==" % (time.time() - start_time))
    #print("==>F1_score=",f,"<==")


# In[29]:


import matplotlib.pyplot as plt


# Define the labels
labels = ['Accuracy 1', 'Accuracy 2']
accuracies = [accuracy1, accuracy2]

# Plot the bar graph
plt.figure(figsize=(5, 3))
plt.bar(labels, accuracies, color=['blue', 'yellow'])
plt.ylim(0, 1)  # Set y-axis limit
plt.ylabel('Accuracy')
plt.title('Comparison of Two Accuracies')
plt.show()


# In[ ]:




