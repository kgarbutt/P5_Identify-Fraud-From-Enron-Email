#!/usr/bin/python
#poi_id.py

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np

import matplotlib
# %matplotlib inline ### needed for Juypter

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

features_list = ['poi',
				'bonus',
				#'deferral_payments',
				'deferred_income',
				#'director_fees',
				'exercised_stock_options',
				#'expenses',
				#'loan_advances',
				#'long_term_incentive',
				#'restricted_stock',
				#'restricted_stock_deferred',
				'salary',
				#'total_payments',
				'total_stock_value',
				#'from_messages',
				#'from_poi_to_this_person',
				#'from_this_person_to_poi',
				#'shared_receipt_with_poi',
				#'to_messages',
				'bonus_salary_ratio']
				#'total_exercised_ratio' ### Added 2 new features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



### Task 2: Remove outliers

### get length before removing outliers
print 'length before removing', len(data_dict)

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers:
    data_dict.pop(outlier, 0)
    
### get length after removing outliers
print 'length after removing', len(data_dict)



### Task 3: Create new feature(s)
### New features are bonus_salary_ratio and total_exercised_ratio

# bonus-salary ratio
for employee, features in data_dict.iteritems():
	if features['bonus'] == "NaN" or features['salary'] == "NaN":
		features['bonus_salary_ratio'] = "NaN"
	else:
		features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])
    
# total_stock_value-excised_stock_options ratio
for stock_holder, features in data_dict.iteritems():
	if features['total_stock_value'] == "NaN" or features['exercised_stock_options'] == "NaN":
		features['total_exercised_ratio'] = "NaN"
	else:
		features['total_exercised_ratio'] = float(features['total_stock_value']) / float(features['exercised_stock_options'])



### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Used only to score the features
### Get kbest features
######################
#
#from sklearn.feature_selection import SelectKBest
#
### Remove poi form feature list
#features_list_wo_pio = list(features_list)
#features_list_wo_pio.remove('poi')
#
### Fit k_best to features and report scores
#k_best = SelectKBest(k="all").fit(features, labels)
#print "k best", sorted(k_best.scores_, reverse=True)
#print "Features sorted by score:", [features_list_wo_pio[i] for i in np.argsort(k_best.scores_)[::-1]]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV ### parameter tuner

#from sklearn.decomposition import PCA
#from sklearn.preprocessing import MinMaxScaler


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### StandardScaling
###################

from sklearn.preprocessing import StandardScaler

# Rescale the data to zero mean and unit variance
scaler = StandardScaler().fit(features_train)

# Transform data
features_train_scaled = scaler.transform(features_train)

features_test_scaled = scaler.transform(features_test)

# Print dataset properties before and after scaling
#print 'transformed shape: {}'.format(features_train_scaled.shape)

### After scaling/transforming
#print 'Training set after scaling:'
#print 'per-feature minimum after scaling:\n {}'.format(features_train_scaled.min(axis=0))
#print 'per-feature maximum after scaling:\n {}'.format(features_train_scaled.max(axis=0))

#print 'Test set after scaling:'
#print 'per-feature minimum after scaling:\n {}'.format(features_test_scaled.min(axis=0))
#print 'per-feature maximum after scaling:\n {}'.format(features_test_scaled.max(axis=0))


### GaussianNB()

### Create and fit the classifier with training dataset
clf = GaussianNB()
clf = clf.fit(features_train_scaled, labels_train)

### Use the trained classifier to predict the labels for the test features
g_pred = clf.predict(features_test_scaled)

### Function for calculating fraction of correct predictions (TP+TN) from all samples (TP+TN+FP+FN)
print 'g_accuracy: ', clf.score(features_test_scaled, labels_test)

### Function for calculating ratio of true positives (TP) out of all positives (TP+FP)
print 'g_precision_score: ', precision_score(labels_test, g_pred, average = 'binary')

### Function for calculating fraction of true predictions from from positive predictions (TP+FN)
print 'g_recall_score: ', recall_score(labels_test, g_pred, average = 'binary')

### Function for calculating the weighted average of the precision and recall scores
print 'g_f1_score: ', f1_score(labels_test, g_pred, average = 'binary')

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
