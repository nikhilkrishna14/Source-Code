#!/usr/bin/env python
# coding: utf-8

# # Ddos attack detection by feature selction with decision tree algorithm
# 
# ## Overall method description
# 
# ### Data preprocessing:
# Onehot encoding is used to make all features of predictors numerical. Further scaling of features are done for avoiding the large values during classfication which can affect the results.
# 
# ### Feature Selection:
# Elimination of redundant and irrelevant data by selecting a segment of important features that approximately entirely represents the data. Univariate feature selection is performed here with the help of Analysis of Variance with F-test. By this method the strength of correlation between the labels in class and features of the categorical variables are determined and then by the SecondPercentile method the features with highest percentile scores are selected to fit with decision tree model. After selecting the segment with important features Recursive Feature Elimination is performed.
# 
# ### Fitting model to training set:
# The conventional decision tree algorithm is applied.
# 
# ### Evaulation of model by comparison of predicted and actual labels of class:
# Evaluation is performed on the test set which is taken randomly as 20% of the entire dataset.
# The evaluation metrics are:accuracy score, recall, f-measure, confusion matrix.
# CV= 10 is considered for cross-validation

# ## Libaries import

# In[1]:


import pandas as pd
import numpy as np
import sys
import sklearn
import time


# ## Load the Dataset

# In[2]:


# Variable or column names of the dataset

var_titles = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


df = pd.read_csv("KDDdataset.csv", header=None, names = var_titles)
train = df.sample(frac=0.8,random_state=200) # setting random state for reproducibility and selecting 80% of data for training
df_test = df.drop(train.index) # selecting rest 20% as test set
df = train
# size and shape of training and test data
print('Training set shape:',df.shape)
print('Test set shape:',df_test.shape)


# ## first 5 instances of training data

# In[3]:


# first five rows
df.head(5)


# ## Descriptive statistics

# In[4]:


df.describe()


# ## Labels in both sets

# In[5]:


print('Labels in Training set:')
print(df['label'].value_counts())
print()
print('Labels in Test set:')
print(df_test['label'].value_counts())


# # Data preprocessing:

# ## Categorical features detection

# In[6]:


# colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
# explore categorical features
print('Training set:')
for var_title in df.columns:
    if df[var_title].dtypes == 'object' :
        unique_cat = len(df[var_title].unique())
        print("Feature '{var_title}' has {unique_cat} categories".format(var_title=var_title, unique_cat=unique_cat))

#see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())


# In[7]:


# Test set
print('Test set:')
for var_title in df_test.columns:
    if df_test[var_title].dtypes == 'object' :
        unique_cat = len(df_test[var_title].unique())
        print("Feature '{var_title}' has {unique_cat} categories".format(var_title=var_title, unique_cat=unique_cat))


# ### Dummy variables are intriduced as the distribution is fairly even.
# ### Test set shows that it has less categories than training set and thus some categories are added to test set.

# # LabelEncoder

# ### Introducing categorical features

# In[8]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
cat_cols=['protocol_type', 'service', 'flag']
# inserting categorical values into cat_cols
cat_cols=['protocol_type', 'service', 'flag'] 
cat_vals_df = df[cat_cols]
testcat_vals_df = df_test[cat_cols]
cat_vals_df.head()


# ### Make column names for dummies

# In[9]:


# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
# combining
dumcols=unique_protocol2 + unique_service2 + unique_flag2
print(dumcols)

# similarly for test set
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2


# ## Transformation of categorical features by LabelEncoder()

# In[10]:


cat_vals_df_enc=cat_vals_df.apply(LabelEncoder().fit_transform)
print(cat_vals_df_enc.head())
# test set
testcat_vals_df_enc=testcat_vals_df.apply(LabelEncoder().fit_transform)


# # One-Hot-Encoding

# In[11]:


enc = OneHotEncoder(sparse= False)
cat_vals_df = enc.fit_transform(cat_vals_df_enc[['protocol_type','service','flag']])
df_cat_data = pd.DataFrame(cat_vals_df,columns=dumcols)
df_cat_data.head()
# df_cat_data = pd.DataFrame(cat_vals_df_enc,columns=dumcols)
# print(df_cat_data.head())
# # # test set
testcat_vals_df = enc.fit_transform(testcat_vals_df_enc[['protocol_type','service','flag']])
testdf_cat_data = pd.DataFrame(testcat_vals_df,columns=testdumcols)

print(df_cat_data.head())
print(testdf_cat_data.head())


# ### Addtion of 6 missing categories to test set which is present in training set

# In[12]:


service_atk_train=df['service'].tolist()
service_atk_test= df_test['service'].tolist()
difference=list(set(service_atk_train) - set(service_atk_test))
string = 'service_'
difference=[string + x for x in difference]
difference


# In[13]:


for col in difference:
    testdf_cat_data[col] = 0

testdf_cat_data.shape


# ## Joining categorical and non-categorical dataframe

# In[14]:


newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
# test data
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)
print(newdf.shape)
print(newdf_test.shape)


# # Splitting Dataset into 4 datasets for every attack category
# ## Renaming every attack label: 0=normal, 1=DoS, 2=Probe, 3=R2L and 4=U2R.
# ## Replacing labels column with new labels column
# ## Make new datasets
# 

# In[15]:


# take label column
labeldf=newdf['label']
labeldf_test=newdf_test['label']
# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test
print(newdf['label'].head())


# In[16]:


to_drop_DoS = [2,3,4]
to_drop_Probe = [1,3,4]
to_drop_R2L = [1,2,4]
to_drop_U2R = [1,2,3]
DoS_df=newdf[~newdf['label'].isin(to_drop_DoS)];
Probe_df=newdf[~newdf['label'].isin(to_drop_Probe)];
R2L_df=newdf[~newdf['label'].isin(to_drop_R2L)];
U2R_df=newdf[~newdf['label'].isin(to_drop_U2R)];

#test
DoS_df_test=newdf_test[~newdf_test['label'].isin(to_drop_DoS)];
Probe_df_test=newdf_test[~newdf_test['label'].isin(to_drop_Probe)];
R2L_df_test=newdf_test[~newdf_test['label'].isin(to_drop_R2L)];
U2R_df_test=newdf_test[~newdf_test['label'].isin(to_drop_U2R)];
print('Train:')
print('Dimensions of DoS:' ,DoS_df.shape)
print('Dimensions of Probe:' ,Probe_df.shape)
print('Dimensions of R2L:' ,R2L_df.shape)
print('Dimensions of U2R:' ,U2R_df.shape)
print('Test:')
print('Dimensions of DoS:' ,DoS_df_test.shape)
print('Dimensions of Probe:' ,Probe_df_test.shape)
print('Dimensions of R2L:' ,R2L_df_test.shape)
print('Dimensions of U2R:' ,U2R_df_test.shape)


# # Feature Scaling:

# In[17]:


# Split dataframes into X & Y
# assign X as a dataframe of feautures and Y as a series of outcome variables
X_DoS = DoS_df.drop('label',1)
Y_DoS = DoS_df.label
X_Probe = Probe_df.drop('label',1)
Y_Probe = Probe_df.label
X_R2L = R2L_df.drop('label',1)
Y_R2L = R2L_df.label
X_U2R = U2R_df.drop('label',1)
Y_U2R = U2R_df.label
# test set
X_DoS_test = DoS_df_test.drop('label',1)
Y_DoS_test = DoS_df_test.label
X_Probe_test = Probe_df_test.drop('label',1)
Y_Probe_test = Probe_df_test.label
X_R2L_test = R2L_df_test.drop('label',1)
Y_R2L_test = R2L_df_test.label
X_U2R_test = U2R_df_test.drop('label',1)
Y_U2R_test = U2R_df_test.label


# ### Save a list of feature names for later use (it is the same for every attack category). Column names are dropped at this stage.

# In[18]:


colNames=list(X_DoS)
colNames_test=list(X_DoS_test)
X_DoS = X_DoS.fillna(0)
Y_DoS = Y_DoS.fillna(0)
X_Probe = X_Probe.fillna(0)
Y_Probe = Y_Probe.fillna(0)
X_R2L = X_R2L.fillna(0)
Y_R2L = Y_R2L.fillna(0)
X_U2R = X_U2R.fillna(0)
Y_U2R = Y_U2R.fillna(0)

X_DoS_test = X_DoS_test.fillna(0)
Y_DoS_test = Y_DoS_test.fillna(0)
X_Probe_test = X_Probe_test.fillna(0)
Y_Probe_test = Y_Probe_test.fillna(0)
X_R2L_test = X_R2L_test.fillna(0)
Y_R2L_test = Y_R2L_test.fillna(0)
X_U2R_test = X_U2R_test.fillna(0)
Y_U2R_test = Y_U2R_test.fillna(0)


# ## Use StandardScaler() to scale the dataframes

# In[19]:


from sklearn import preprocessing
scaler1 = preprocessing.StandardScaler().fit(X_DoS)
X_DoS=scaler1.transform(X_DoS) 
scaler2 = preprocessing.StandardScaler().fit(X_Probe)
X_Probe=scaler2.transform(X_Probe) 
scaler3 = preprocessing.StandardScaler().fit(X_R2L)
X_R2L=scaler3.transform(X_R2L) 
scaler4 = preprocessing.StandardScaler().fit(X_U2R)
X_U2R=scaler4.transform(X_U2R) 
# test data
scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)
X_DoS_test=scaler5.transform(X_DoS_test) 
scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)
X_Probe_test=scaler6.transform(X_Probe_test) 
scaler7 = preprocessing.StandardScaler().fit(X_R2L_test)
X_R2L_test=scaler7.transform(X_R2L_test) 
scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)
X_U2R_test=scaler8.transform(X_U2R_test) 


# ### Check that the Standard Deviation is 1

# In[20]:


print(X_DoS.std(axis=0))


# In[21]:


print(X_Probe.std(axis=0))


# In[22]:


print(X_R2L.std(axis=0))


# In[23]:


print(X_U2R.std(axis=0))


# # Feature Selection:

# # 1. Univariate Feature Selection using ANOVA F-test

# In[24]:


#univariate feature selection with ANOVA F-test. using secondPercentile method, then RFE
#Scikit-learn exposes feature selection routines as objects that implement the transform method
#SelectPercentile: removes all but a user-specified highest scoring percentage of features
#f_classif: ANOVA F-value between label/feature for classification tasks.
from sklearn.feature_selection import SelectPercentile, f_classif
np.seterr(divide='ignore', invalid='ignore');
selector=SelectPercentile(f_classif, percentile=10)
X_newDoS = selector.fit_transform(X_DoS,Y_DoS)
print(X_DoS.shape)
print(Y_DoS.shape)
print(X_newDoS.shape)


# ### Get the features that were selected: DoS

# In[25]:


true=selector.get_support()
newcolindex_DoS=[i for i, x in enumerate(true) if x]
newcolname_DoS=list( colNames[i] for i in newcolindex_DoS )
newcolname_DoS


# In[26]:


print(X_Probe.shape)
print(Y_Probe.shape)
X_newProbe = selector.fit_transform(X_Probe,Y_Probe)
X_newProbe.shape


# ### Get the features that were selected: Probe

# In[27]:


true=selector.get_support()
newcolindex_Probe=[i for i, x in enumerate(true) if x]
newcolname_Probe=list( colNames[i] for i in newcolindex_Probe )
newcolname_Probe


# In[28]:


X_newR2L = selector.fit_transform(X_R2L,Y_R2L)
X_newR2L.shape


# ### Get the features that were selected: R2L

# In[29]:


true=selector.get_support()
newcolindex_R2L=[i for i, x in enumerate(true) if x]
newcolname_R2L=list( colNames[i] for i in newcolindex_R2L)
newcolname_R2L


# In[30]:


X_newU2R = selector.fit_transform(X_U2R,Y_U2R)
X_newU2R.shape


# ### Get the features that were selected: U2R

# In[31]:


true=selector.get_support()
newcolindex_U2R=[i for i, x in enumerate(true) if x]
newcolname_U2R=list( colNames[i] for i in newcolindex_U2R)
newcolname_U2R


# # Summary of features selected by Univariate Feature Selection

# In[32]:


print('Features selected for DoS:',newcolname_DoS)
print()
print('Features selected for Probe:',newcolname_Probe)
print()
print('Features selected for R2L:',newcolname_R2L)
print()
print('Features selected for U2R:',newcolname_U2R)


# ## As a second option RFE is considered where 13 features are selected for each category of attack.

# # 2. Recursive Feature Elimination for feature ranking (Option 1: get importance from previous selected)

# In[33]:


from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
# Create a decision tree classifier. By convention, clf means 'classifier'
clf = DecisionTreeClassifier(random_state=0)

#rank all features, i.e continue the elimination until the last one
start_time = time.time()
rfe = RFE(clf, n_features_to_select=1)
rfe.fit(X_newDoS, Y_DoS)
end_time = time.time()
print ("DoS Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_DoS)))
print('Training time to fit classifier to DoS attack type data is',end_time-start_time,'secs')


# In[34]:


start_time = time.time()
rfe.fit(X_newProbe, Y_Probe)
end_time = time.time()
print ("Probe Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_Probe)))
print('Training time to fit classifier to Probe attack type data is',end_time-start_time,'secs')


# In[35]:


start_time = time.time()
rfe.fit(X_newR2L, Y_R2L)
end_time = time.time()
print ("R2L Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_R2L)))
print('Training time to fit classifier to R2L attack type data is',end_time-start_time,'secs')


# In[36]:


start_time = time.time()
rfe.fit(X_newU2R, Y_U2R)
end_time = time.time()
print ("U2R Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), newcolname_U2R)))
print('Training time to fit classifier to U2R attack type data is',end_time-start_time,'secs')


# # 2. Recursive Feature Elimination, select 13 features each of 122 (Option 2: get 13 best features from 122 from RFE)

# In[37]:


from sklearn.feature_selection import RFE
clf = DecisionTreeClassifier(random_state=0)
rfe = RFE(estimator=clf, n_features_to_select=13, step=1)
rfe.fit(X_DoS, Y_DoS)
X_rfeDoS=rfe.transform(X_DoS)
true=rfe.support_
rfecolindex_DoS=[i for i, x in enumerate(true) if x]
rfecolname_DoS=list(colNames[i] for i in rfecolindex_DoS)


# In[38]:


rfe.fit(X_Probe, Y_Probe)
X_rfeProbe=rfe.transform(X_Probe)
true=rfe.support_
rfecolindex_Probe=[i for i, x in enumerate(true) if x]
rfecolname_Probe=list(colNames[i] for i in rfecolindex_Probe)


# In[39]:


rfe.fit(X_R2L, Y_R2L)
X_rfeR2L=rfe.transform(X_R2L)
true=rfe.support_
rfecolindex_R2L=[i for i, x in enumerate(true) if x]
rfecolname_R2L=list(colNames[i] for i in rfecolindex_R2L)


# In[40]:


rfe.fit(X_U2R, Y_U2R)
X_rfeU2R=rfe.transform(X_U2R)
true=rfe.support_
rfecolindex_U2R=[i for i, x in enumerate(true) if x]
rfecolname_U2R=list(colNames[i] for i in rfecolindex_U2R)


# # Summary of features selected by RFE

# In[41]:


print('Features selected for DoS:',rfecolname_DoS)
print()
print('Features selected for Probe:',rfecolname_Probe)
print()
print('Features selected for R2L:',rfecolname_R2L)
print()
print('Features selected for U2R:',rfecolname_U2R)


# In[42]:


print(X_rfeDoS.shape)
print(X_rfeProbe.shape)
print(X_rfeR2L.shape)
print(X_rfeU2R.shape)


# # Step 4: Build the model:
# ### Classifier is trained for all features and for reduced features, for later comparison.
# #### The classifier model itself is stored in the clf variable.

# In[43]:


# all features
clf_DoS=DecisionTreeClassifier(random_state=0)
clf_Probe=DecisionTreeClassifier(random_state=0)
clf_R2L=DecisionTreeClassifier(random_state=0)
clf_U2R=DecisionTreeClassifier(random_state=0)
clf_DoS.fit(X_DoS, Y_DoS)
clf_Probe.fit(X_Probe, Y_Probe)
clf_R2L.fit(X_R2L, Y_R2L)
clf_U2R.fit(X_U2R, Y_U2R)


# In[44]:


# selected features
clf_rfeDoS=DecisionTreeClassifier(random_state=0)
clf_rfeProbe=DecisionTreeClassifier(random_state=0)
clf_rfeR2L=DecisionTreeClassifier(random_state=0)
clf_rfeU2R=DecisionTreeClassifier(random_state=0)
clf_rfeDoS.fit(X_rfeDoS, Y_DoS)
clf_rfeProbe.fit(X_rfeProbe, Y_Probe)
clf_rfeR2L.fit(X_rfeR2L, Y_R2L)
clf_rfeU2R.fit(X_rfeU2R, Y_U2R)


# # Evaulation of model by comparison of predicted and actual labels of class:

# # Using all Features for each category

# # Confusion Matrices
# ## DoS

# In[45]:


# Apply the classifier we trained to the test data (which it has never seen before)
clf_DoS.predict(X_DoS_test)


# In[46]:


# View the predicted probabilities of the first 10 observations
clf_DoS.predict_proba(X_DoS_test)[0:10]


# In[47]:


Y_DoS_pred=clf_DoS.predict(X_DoS_test)
# Create confusion matrix
pd.crosstab(Y_DoS_test, Y_DoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])


# # Cross Validation: Accuracy, Precision, Recall, F-measure

# ## DoS

# In[48]:


from sklearn.model_selection import cross_val_score
from sklearn import metrics
accuracy = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))


# # RFECV for visualization

# In[49]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


##
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
##
### Create the RFE object and compute a cross-validated score.
###The "accuracy" scoring is proportional to the number of correct
### classifications
rfecv_DoS = RFECV(estimator=clf_DoS, step=1, cv=10, scoring='accuracy')
rfecv_DoS.fit(X_DoS_test, Y_DoS_test)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.title('RFECV DoS')
plt.plot(range(1, len(rfecv_DoS.grid_scores_) + 1), rfecv_DoS.grid_scores_)
plt.show()


# # DoS attack classfication with selected 13 features

# # Confusion Matrices
# ## DoS

# In[51]:


# reduce test dataset to 13 features, use only features described in rfecolname_DoS etc.
X_DoS_test2=X_DoS_test[:,rfecolindex_DoS]
X_Probe_test2=X_Probe_test[:,rfecolindex_Probe]
X_R2L_test2=X_R2L_test[:,rfecolindex_R2L]
X_U2R_test2=X_U2R_test[:,rfecolindex_U2R]
X_U2R_test2.shape


# In[52]:


Y_DoS_pred2=clf_rfeDoS.predict(X_DoS_test2)
# Create confusion matrix
pd.crosstab(Y_DoS_test, Y_DoS_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])


# # Cross Validation: Accuracy, Precision, Recall, F-measure

# ## DoS

# In[53]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))


# # Stratified Cross fold or number of folds is constant which is considered 10

# In[54]:


from sklearn.model_selection import StratifiedKFold
accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=StratifiedKFold(10), scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))


# # Cross validation with 3,10,15,25 and 40 folds

# ## DoS

# In[55]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=3, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=3, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=3, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=3, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))


# In[56]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))


# In[57]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=15, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=15, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=15, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=15, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))


# In[58]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=25, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=25, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=25, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=25, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))


# In[59]:


accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=40, scoring='accuracy')
print("Accuracy: %0.5f (+/- %0.5f)" % (accuracy.mean(), accuracy.std() * 2))
precision = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=40, scoring='precision')
print("Precision: %0.5f (+/- %0.5f)" % (precision.mean(), precision.std() * 2))
recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=40, scoring='recall')
print("Recall: %0.5f (+/- %0.5f)" % (recall.mean(), recall.std() * 2))
f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=40, scoring='f1')
print("F-measure: %0.5f (+/- %0.5f)" % (f.mean(), f.std() * 2))


# In[61]:


import matplotlib.pyplot as plt


x1 = [0.99878, 0.99922, 0.99922, 0.99911, 0.99923 ]
x2 = [0.99867, 0.99856, 0.99867, 0.99867, 0.99878 ]

y = [3, 10,  15,  25, 40]
plt.ylim(0.998,1)
plt.plot(y, x1)
plt.plot(y, x2)
plt.legend

plt.xlabel('cross validation parameter values')
plt.ylabel('Precisio/Recall')

plt.title('DoS')

plt.show()


# In[9]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('DoS')
y_pos = objects
performance = [0.99920,0.99916,0.99930,0.99928,0.99948]
performance1=[0.99899,0.99912,0.99916,0.99912 ,0.99921]



import matplotlib.pyplot as plt

y1=10;


y = [3, 10,  15,  25, 40]
plt.ylim(0.998,1)
plt.plot(y, performance)
plt.plot(y, performance1)
plt.legend

plt.xlabel('cross validation parameter values')
plt.ylabel('Accuracy')

plt.title('DoS')

plt.show()


# In[8]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('DoS')
y_pos = objects
performance = [2.5202605724334717]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos)
plt.xlabel('Attacks')
plt.ylabel('Training time to fit classification in Secs')
plt.title('Training Time of DoS')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




