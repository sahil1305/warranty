#importing libraries
import pandas as pd #for dataframes
import numpy as np #for numerical operations
import matplotlib.pyplot as plt #for visualization
import seaborn as sb #for plottings
from sklearn import preprocessing #for preprossing the data
from sklearn.utils import resample #for upsampling and downsampling
from sklearn.model_selection import train_test_split #for splitting into train and test
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#importing the dataset
claims= pd.read_excel("H:\\all datasets\\Claims.xlsx")

#information of the dataset
claims.info()

# Deleting first column
claims.drop(["Serial"],inplace=True,axis=1)     

#finding missing values in the data
claims.isnull().sum() #here the missing values is in CLaim_Value i.e 333

#getting bar plots and actudal counts of the categorical columns
sb.countplot(x="Region",data=claims)
claims.Region.value_counts()
sb.countplot(x="State",data=claims)
claims.State.value_counts()
sb.countplot(x="Area",data=claims)
claims.Area.value_counts()
sb.countplot(x="City",data=claims)
claims.City.value_counts()
sb.countplot(x="Consumer_profile",data=claims)
claims.Consumer_profile.value_counts()
sb.countplot(x="Product_category",data=claims)
claims.Product_category.value_counts()
sb.countplot(x="Product_type",data=claims)
claims.Product_type.value_counts()
sb.countplot(x="Purchased_from",data=claims)
claims.Purchased_from.value_counts()
sb.countplot(x="Purpose",data=claims)
claims.Purpose.value_counts()

sb.countplot(x="Fraud",data=claims)
claims.Fraud.value_counts()
sb.countplot(x="Service_Centre",data=claims)
claims.Service_Centre.value_counts()
sb.countplot(x="AC_1001_Issue",data=claims)
claims.AC_1001_Issue.value_counts()
sb.countplot(x="AC_1002_Issue",data=claims)
claims.AC_1002_Issue.value_counts()
sb.countplot(x="AC_1003_Issue",data=claims)
claims.AC_1003_Issue.value_counts()
sb.countplot(x="TV_2001_Issue",data=claims)
claims.TV_2001_Issue.value_counts()
sb.countplot(x="TV_2002_Issue",data=claims)
claims.TV_2002_Issue.value_counts()
sb.countplot(x="TV_2003_Issue",data=claims)
claims.TV_2003_Issue.value_counts()

#histogram of continous variables
claims.Product_Age.plot.hist()
claims.Claim_Value.plot.hist()
claims.Call_details.plot.hist()

#pair plots
pd.crosstab(claims.State,claims.Fraud).plot(kind="bar")
pd.crosstab(claims.Region,claims.Fraud).plot(kind="bar")
pd.crosstab(claims.City,claims.Fraud).plot(kind="bar")
pd.crosstab(claims.Area,claims.Fraud).plot(kind="bar")
pd.crosstab(claims.Consumer_profile,claims.Fraud).plot(kind="bar")
pd.crosstab(claims.Product_type,claims.Fraud).plot(kind="bar")
pd.crosstab(claims.Purchased_from,claims.Fraud).plot(kind="bar")
pd.crosstab(claims.Purpose,claims.Fraud).plot(kind="bar")
pd.crosstab(claims.Service_Centre,claims.Fraud).plot(kind="bar")

#merging UP with Uttar Pradesh in State column
claims.loc[(claims.State == "UP"), "State"] = "Uttar Pradesh"

#merging claim with Claim in Purpose column
claims.loc[(claims.Purpose == "claim"), "Purpose"] = "Claim"

#Separating hyderbad among two states as it is there in both Telangna and Andra Pradesh. 
#For Andhra Pradesh = Hyderbad, Telengana = Hyderabad 1
claims.loc[(claims.State == "Telengana"), "City"] = "Hyderabad 1"

#region correction according to states
claims.loc[(claims.State == "Delhi") | (claims.State == "Uttar Pradesh") |
        (claims.State == "Haryana") | (claims.State == "HP") | (claims.State == "J&K"), "Region"] = "North"

claims.loc[(claims.State == "Andhra Pradesh") | (claims.State == "Karnataka") |
        (claims.State == "Kerala")  | (claims.State == "Tamilnadu") | 
        (claims.State == "Telengana"), "Region"] = "South"
        
 claims.loc[(claims.State == "Assam") | 
        (claims.State == "Tripura") | (claims.State == "West Bengal"), "Region"] = "East"

claims.loc[(claims.State == "Gujarat"), "Region"] = "West"

claims.loc[(claims.State == "Bihar") | (claims.State == "Jharkhand") | (claims.State == "Odisha"), "Region"] = "North East"

claims.loc[(claims.State == "Goa") | (claims.State == "Maharshtra"), "Region"] = "South West"

claims.loc[(claims.State == "Rajasthan"), "Region"] = "North West"

claims.loc[(claims.State == "MP"), "Region"] = "Central"

#getting the duplicate values
claims.Claim_Value.duplicated().sum()

#deleting the duplicate values
claims1=claims.drop_duplicates()

#missing values of new data i.e claims1
claims1.isnull().sum() #there are 9 missing values in the Claim_Value column

#boxplot of the Claim_value column to find if there is any outliers
sb.boxplot(x="Claim_Value",data=claims1) #it consists outliers

#median for the missing values
claims1["Claim_Value"].median() #7370

#median imputation 
claims1["Claim_Value"].fillna(7370,inplace=True) #median of Claim_Value is 7370 

#checking if any missing values in dataset
claims1.isnull().sum()

#checking the output column
sb.countplot(x="Fraud",data=claims1)
claims1.Fraud.value_counts() #the dataset is imbalance i.e 323= genuine and 35=fraud

#creating dummies for categorical variables
dummies = pd.get_dummies(claims1[['Region','State','Area','City','Consumer_profile','Product_category','Product_type',
                                  'Purchased_from','Purpose']])
list(dummies) #list of columns

#Dropping the columns for which we have created dummies
claims1.drop(['Region','State','Area','City','Consumer_profile','Product_category','Product_type',
             'Purchased_from','Purpose'],inplace=True,axis = 1)
 
# adding the dummies and claims1 column 
claims2 = pd.concat([claims1,dummies],axis=1)

#shape of the dataset
claims2.shape
claims2['Fraud'].value_counts()

# Separate majority and minority classes
claims2_majority = claims2[claims2.Fraud==0]
claims2_minority = claims2[claims2.Fraud==1]

# Upsample minority class
claims2_minority_upsampled = resample(claims2_minority,replace=True,n_samples=323,random_state=123) #sample with replacement
                                                                                                    #to match majority class        
                                                                                                    #reproducible results        
                                                                                                    
# Combine majority class with upsampled minority class
claims2_upsampled = pd.concat([claims2_majority, claims2_minority_upsampled])

# Display new class counts
claims2_upsampled.Fraud.value_counts()                                                                                                     

#first making model on the claims2_upsampled data
x= claims2_upsampled.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81]]
y= claims2_upsampled.iloc[:,[10]]

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#predict the response for train dataset
y_pred1 = clf.predict(x_train)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train, y_pred1))

print ('\n clasification report:\n', metrics.classification_report(y_test,y_pred))

#confusion matrix
print ('\n confussion matrix:\n',metrics.confusion_matrix(y_test,y_pred))


#downsample majority class
claims2_majority_downsampled = resample(claims2_majority,replace=False,n_samples=35,random_state=123)

#combine minority class with downsampled majority class
claims2_downsampled = pd.concat([claims2_minority,claims2_majority_downsampled])

#display new class counts
claims2_downsampled.Fraud.value_counts()

#let's make the model on claims2_downsampled data
x1= claims2_downsampled.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81]]
y1= claims2_downsampled.iloc[:,[10]]

# Split dataset into training set and test set
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, random_state=123)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x1_train,y1_train)

#Predict the response for test dataset
y1_pred = clf.predict(x1_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y1_test, y1_pred))

#predict the response for train dataset
y1_pred1 = clf.predict(x1_train)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y1_train, y1_pred1))

print ('\n clasification report:\n', metrics.classification_report(y1_test,y1_pred))

#confusion matrix
print ('\n confussion matrix:\n',metrics.confusion_matrix(y1_test,y1_pred))



#from the above upsampled and downsampled we conclude that upsampled data 
#is giving good accuracy of 94% and downsampled data is giving accuracy of 57% while 
#building the decision tree classifier model 
#So, we finaize the upsampled data for model building.


#logistics regression
#Create logistic regression classifier object
logreg = LogisticRegression()
#train logistic regression classifier
logreg= logreg.fit(x_train, y_train)
#Predict the response for test dataset
y_pred = logreg.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#predict the response for train dataset
y_pred1 = logreg.predict(x_train)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train, y_pred1))
print ('\n clasification report:\n', metrics.classification_report(y_test,y_pred))
#confusion matrix
print ('\n confussion matrix:\n',metrics.confusion_matrix(y_test,y_pred))


#Random Forest model
random_forest = RandomForestClassifier()
random_forest= random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#predict the response for train dataset
y_pred1 = random_forest.predict(x_train)
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_train, y_pred1))
print ('\n clasification report:\n', metrics.classification_report(y_test,y_pred))
#confusion matrix
print ('\n confussion matrix:\n',metrics.confusion_matrix(y_test,y_pred))

