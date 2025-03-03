# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import StandardScaler



# 2. Create test set if you like to do the split programmatically or if you have not already split the data at this point
#Loaded the dataset disadvantaged_communities and assign to dataset
dataset = pd.read_csv('disadvantaged_communities.csv')

#Cleans the dataset by finding any if any values are missing and fills it with the mean of the column
dataset.fillna(dataset.mean(), inplace = True)

#Assigns the ordinal encoder to enc
enc = OrdinalEncoder()
#Encodes the categorical variables in the dataset and assigns those encoded variables to where the categorical values where in the dataset
dataset = enc.fit_transform(dataset)

#Splits the dataset into X: Predictor variables from y: Target variable which is CES 4.0 Percentile Range
X = dataset.drop('CES 4.0 Percentile Range', axis=1)
y = dataset['CES 4.0 Percentile Range']

#Trains and test the dataset with a 80% 20% split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0, stratify = y)

#Assigns the standard scaler to sc
sc = StandardScaler()

#Standardizes and transforms X_test
X_test = sc.fit_transform(X_test)

# Below is the code to convert X_test into data frames for the next steps
cols = X.columns
X_test = pd.DataFrame(X_test, columns = cols) # pd is the imported pandas lirary - Import pandas as pd



# 3. Load your saved model for dissadvantaged communities classification 
#that you saved in dissadvantaged_communities_classification.py via Pikcle
#Loads our trained SVM classifier model from the DACs_classification file
with open('SvmClassifier.sav', 'rb') as f:
    SVMclassifier = pk.load(f)
    
#Loads our trained RF classifier model from the DACs_classification file
with open('RfClassifier.sav', 'rb') as f:
    RF_Classifier = pk.load(f)



# 4. Make predictions on test_set created from step 2
#Test the SVM classifier with X_test and assigns it to SVM_y_pred
SVM_y_pred = SVMclassifier.predict(X_test)
#Test the RF classifier with X_test and assigns it to RF_y_pred
RF_y_pred = RF_Classifier.predict(X_test)



# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.

#SVM Classifier
# Get and print confusion matrix
#Creates the confusion matrix of the SVM and assigns it to SVM_cm then prints it
SVM_cm = confusion_matrix(y_test, SVM_y_pred)
print('SVM Confusion Matrix: \n', SVM_cm)
#Computes and prints the accuracy score of the SVM
print('SVM Accuracy Score: ', accuracy_score(y_test, SVM_y_pred))

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

# Compute Recall and use the following line to print it
#Computes the recall and prints it
recall = SVM_TP / (SVM_TP + SVM_FN)
print('SVM Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
#Computes the specificity and prints it
specificity = SVM_TN / (SVM_TN + SVM_FP)
print('SVM Specificity : {0:0.3f}'.format(specificity))


#Random Forest Classifier
#Creates the confusion matrix of the RF and assigns it to RF_cm then prints it
RF_cm = confusion_matrix(y_test, RF_y_pred)
print('\nRF Confusion Matrix: \n', RF_cm)
#Computes and prints accuracy score
print('RF Accuracy Score: ', accuracy_score(y_test, RF_y_pred))

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