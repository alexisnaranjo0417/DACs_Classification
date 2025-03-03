#Step 1:
# Import libraries
# In this section, you can use a search engine to look for the functions that will help you implement the following steps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pk
from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier



#Step 2:
# Load dataset and show basic statistics
# 1. Show dataset size (dimensions)
# 2. Show what column names exist for the 49 attributes in the dataset
# 3. Show the distribution of the target class CES 4.0 Percentile Range column
# 4. Show the percentage distribution of the target class CES 4.0 Percentile Range column
#Loaded the dataset disadvantaged_communities and assign to dataset
dataset = pd.read_csv('disadvantaged_communities.csv')


#Prints the size/dimensions of the dataset
print('Data set size: ', dataset.shape, '\n')
#Prints the names of the columns/attributes of the dataset
print('Column Names: \n', dataset.columns.tolist(), '\n')
#Prints the values in the CES 4.0 Percentile Range column
print('CES 4.0 Percentile Range column distribution: \n', dataset['CES 4.0 Percentile Range'].value_counts(), '\n')
#Prints the number of how many times certain values appeared in the column CES 4.0 Percentile Range
print('CES 4.0 Percentile Range column percentage distribution: \n', dataset['CES 4.0 Percentile Range'].value_counts(normalize = True) * 100, '\n')



# Step 3:
#Clean the dataset - you can eitherhandle the missing values in the dataset
# with the mean of the columns attributes or remove rows the have missing values.
#Cleans the dataset by finding any if any values are missing and fills it with the mean of the column
dataset.fillna(dataset.mean(), inplace = True)



# Step 4: 
#Encode the Categorical Variables - Using OrdinalEncoder from the category_encoders library to encode categorical variables as ordinal integers
#Assigns the ordinal encoder to enc
enc = OrdinalEncoder()
#Encodes the categorical variables in the dataset and assigns those encoded variables to where the categorical values where in the dataset
dataset = enc.fit_transform(dataset)



# Step 5: 
# Separate predictor variables from the target variable (attributes (X) and target variable (y) as we did in the class)
# Create train and test splits for model development. Use the 90% and 20% split ratio
# Use stratifying (stratify=y) to ensure class balance in train/test splits
# Name them as X_train, X_test, y_train, and y_test
# Name them as X_train, X_test, y_train, and y_test
#Splits the dataset into X: Predictor variables from y: Target variable which is CES 4.0 Percentile Range
X = dataset.drop('CES 4.0 Percentile Range', axis=1)
y = dataset['CES 4.0 Percentile Range']

#Trains and test the dataset with a 80% 20% split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0, stratify = y)



# Do not do steps 6 - 8 for the Ramdom Forest Model
# Step 6:
# Standardize the features (Import StandardScaler here)
#Assigns the standard scaler to sc
sc = StandardScaler()

#Standardizes and transforms X_train and X_test
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



# Step 7:
# Below is the code to convert X_train and X_test into data frames for the next steps
cols = X.columns
X_train = pd.DataFrame(X_train, columns = cols) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test, columns = cols) # pd is the imported pandas lirary - Import pandas as pd



# Step 8 - Build and train the SVM classifier
# Train SVM with the following parameters. (use the parameters with the highest accuracy for the model)
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3 (Linear)
#Creates the SVM classifier and assigns is to SVMclassifier
SVMclassifier = SVC(kernel = 'rbf', C = 10.0, random_state = 0)
#Trains the SVM classifier with X_train and y_train
SVMclassifier.fit(X_train, y_train)

# Test the above developed SVC on unseen pulsar dataset samples
#Test the SVM classifier with X_test and assigns it to SVM_y_pred
SVM_y_pred = SVMclassifier.predict(X_test)

#Computes and prints the accuracy score of the SVM
print('SVM Accuracy Score: ', accuracy_score(y_test, SVM_y_pred))

# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
#Saves the SVM Classifier model to a file named SvmClassifier.sav
with open('SvmClassifier.sav', 'wb') as f:
   pk.dump(SVMclassifier, f)

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
#Creates the confusion matrix of the SVM and assigns it to SVM_cm then prints it
SVM_cm = confusion_matrix(y_test, SVM_y_pred)
print('SVM Confusion Matrix: \n', SVM_cm)

#Initializes SVM's TP, FP, FN, and TN to 0
SVM_TP_total = 0
SVM_FP_total = 0
SVM_FN_total = 0
SVM_TN_total = 0

#Loops through the confusion matrix of the SVM and adds the TP, FP, FN, and TN of all the classes to a total of TP, FP, FN, and TN
for i in range(len(SVM_cm)):
   #TP total
   SVM_TP_total += SVM_cm[i, i]
   #Saves the TP value for each class to calculate the TN
   SVM_TP_TN_Val = SVM_cm[i, i]
   
   #FP total
   SVM_FP_total += np.sum(SVM_cm[:, i]) - SVM_cm[i, i]
   #Saves the FP value for each class to calculate the TN
   SVM_FP_TN_Val = np.sum(SVM_cm[:, i]) - SVM_cm[i, i]
   
   #FN total
   SVM_FN_total += np.sum(SVM_cm[i, :]) - SVM_cm[i, i]
   #Saves the FN value for each class to calculate the TN
   SVM_FN_TN_Val = np.sum(SVM_cm[i, :]) - SVM_cm[i, i]
   
   #TN total
   SVM_TN_total += np.sum(SVM_cm) - (SVM_TP_TN_Val + SVM_FP_TN_Val + SVM_FN_TN_Val)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
#Takes the total of all the classes TP's and finds the average of them
SVM_TP = SVM_TP_total / len(SVM_cm)
#Takes the total of all the classes TN's and finds the average of them
SVM_TN = SVM_TN_total / len(SVM_cm)
#Takes the total of all the classes FP's and finds the average of them
SVM_FP = SVM_FP_total / len(SVM_cm)
#Takes the total of all the classes FN's and finds the average of them
SVM_FN = SVM_FN_total / len(SVM_cm)

# Compute Precision and use the following line to print it
#Computes the precision score and prints it
precision = SVM_TP / (SVM_TP + SVM_FP)
print('SVM Precision : {0:0.3f}'.format(precision))

# Compute ecall and use the following line to print it
#Computes the recall and prints it
recall = SVM_TP / (SVM_TP + SVM_FN)
print('SVM Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
#Computes the specificity and prints it
specificity = SVM_TN / (SVM_TN + SVM_FP)
print('SVM Specificity : {0:0.3f}'.format(specificity))



# Step 9: Build and train the Random Forest classifier
# Train Random Forest  with the following parameters.
# (n_estimators=10, random_state=0)
#Creates the RF classifier and assigns is to SVMclassifier
RF_classifier = RandomForestClassifier(n_estimators = 10, random_state = 0)
#Trains the RF classifier with X_train and y_train
RF_classifier.fit(X_train, y_train)

# Test the above developed Random Forest model on unseen DACs dataset samples
#Test the RF classifier with X_test and assigns it to RF_y_pred
RF_y_pred = RF_classifier.predict(X_test)

#Computes and prints accuracy score
print('\nRF Accuracy Score: ', accuracy_score(y_test, RF_y_pred))

# Save your Random Forest model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
#Saves the RF Classifier model to a file named RFClassifier.sav
with open('RfClassifier.sav', 'wb') as f:
   pk.dump(RF_classifier, f)

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
#Creates the confusion matrix of the RF and assigns it to RF_cm then prints it
RF_cm = confusion_matrix(y_test, RF_y_pred)
print('RF Confusion Matrix: \n', RF_cm)

#Initializes RF's TP, FP, FN, and TN to 0
RF_TP_total = 0
RF_FP_total = 0
RF_FN_total = 0
RF_TN_total = 0

#Loops through the confusion matrix of the RF and adds the TP, FP, FN, and TN of all the classes to a total of TP, FP, FN, and TN
for i in range(len(RF_cm)):
   #TP total
   RF_TP_total += RF_cm[i, i]
   #Saves the TP value for each class to calculate the TN
   RF_TP_TN_Val = RF_cm[i, i]
   
   #FP total
   RF_FP_total += np.sum(RF_cm[:, i]) - RF_cm[i, i]
   #Saves the FP value for each class to calculate the TN
   RF_FP_TN_Val = np.sum(RF_cm[:, i]) - RF_cm[i, i]
   
   #FN total
   RF_FN_total += np.sum(RF_cm[i, :]) - RF_cm[i, i]
   #Saves the FN value for each class to calculate the TN
   RF_FN_TN_Val = np.sum(RF_cm[i, :]) - RF_cm[i, i]
   
   #TN total
   RF_TN_total += np.sum(RF_cm) - (RF_TP_TN_Val + RF_FP_TN_Val + RF_FN_TN_Val)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
#Takes the total of all the classes TP's and finds the average of them
RF_TP = RF_TP_total / len(RF_cm)
#Takes the total of all the classes TN's and finds the average of them
RF_TN = RF_TN_total / len(RF_cm)
#Takes the total of all the classes FP's and finds the average of them
RF_FP = RF_FP_total / len(RF_cm)
#Takes the total of all the classes FN's and finds the average of them
RF_FN = RF_FN_total / len(RF_cm)

# Compute Classification Accuracy and use the following line to print it
#Computes the classififcation accuracy and prints it
classification_accuracy = (RF_TP + RF_TN) / (RF_TP + RF_TN + RF_FP + RF_FN)
print('RF Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# Compute Precision and use the following line to print it
#Computes the precision score and prints it
precision =  RF_TP / (RF_TP + RF_FP)
print('RF Precision : {0:0.3f}'.format(precision))

# Compute Recall and use the following line to print it
#Computes the recall and prints it
recall =  RF_TP / (RF_TP + RF_FN)
print('RF Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
#Computes the specificity and prints it
specificity = RF_TN / (RF_TN + RF_FP)
print('RF Specificity : {0:0.3f}'.format(specificity))