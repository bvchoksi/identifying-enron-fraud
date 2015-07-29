#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
#from tester import test_classifier, dump_classifier_and_data
from sklearn import cross_validation
import pprint

#Creating functions to be used later
#The classifier function will be used to fit, predict and cross-validate
def classifier(clf, dataset, feature_list):
    #Splitting the dataset into labels and features
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    #Using K-Folds cross-validation to split data into train and test sets
    #Training the classifier on 95% of the data, holding back remaining 5% for testing
    k = 20
    kf = cross_validation.KFold(len(labels), n_folds = k, shuffle = True, random_state = 42)
    
    #For each fold, fitting classifier on the train set, making predictions on the
    #test set, and then comparing the predicted labels against actual labels to
    #calculate precision and recall
    true_positives = 0;
    false_positives = 0;
    true_negatives = 0;
    false_negatives = 0
    
    for train, test in kf:
        labels_train = [labels[i] for i in train]
        features_train = [features[i] for i in train]
        labels_test = [labels[i] for i in test]
        features_test = [features[i] for i in test]
        
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        
        for prediction, actual in zip(predictions, labels_test):
            if prediction == 0 and actual == 0:
                true_negatives += 1
            elif prediction == 0 and actual == 1:
                false_negatives += 1
            elif prediction == 1 and actual == 0:
                false_positives += 1
            else:
                true_positives += 1
    
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        print clf
        print "Accuracy: {:>0.2f}".format(accuracy), \
            "\tPrecision: {:>0.2f}".format(precision), \
            "\tRecall: {:>0.2f}".format(recall), "\r\n"
        print "Total predictions: {0:0d}".format(total_predictions), \
            "\tTrue positives: {0:0d}".format(true_positives), \
            "\tFalse positives: {0:0d}".format(false_positives), \
            "\tTrue negatives: {0:0d}".format(true_negatives), \
            "\tFalse negatives: {0:0d}".format(false_negatives)
    except:
        print "Error in classifier function."

#The dump_classifier_and_data function will be used to create pickle files that will
#contain the classifier, data and feature list
def dump_classifier_and_data(clf, dataset, feature_list):
    pickle.dump(clf, open("my_classifier.pkl", "w") )
    pickle.dump(dataset, open("my_dataset.pkl", "w") )
    pickle.dump(feature_list, open("my_feature_list.pkl", "w") )

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

#First let's explore the dictionary
print "\r\n"
print "DATA DICTIONARY EXPLORATION\r\n"

record_count = len(data_dict.keys())
print "Number of records:", record_count, "\r\n"

keys = [k for k in data_dict[data_dict.keys()[0]]]
keys.sort()
print "Data elements:", keys, "\r\n"

print "Sample values for each data element:"
for k in keys:
    if k != "email_address":
        print k, ":", [data_dict[k1][k] for k1 in data_dict][:10]
print "\r\n"

#email_address is not a feature or label, and hence will be ignored
#poi, which contains a True or False value, indicating POI or not POI, is a label
#The remaining data elements can be used as features with classifiers to predict POI

#Available features fall into two main categories:
	#1. remuneration
	#2. communication

#Remuneration features fall into two sub-categories:
	#1. payments
	#2. stock

#The payment-related remuneration features are components of total_payments
#This was verified using values in the TOTAL key of the dictionary
	#bonus
	#deferral_payments
	#deferred_income
	#director_fees
	#expenses
	#loan_advances
	#long_term_incentive
	#other
	#salary
	#total_payments

#The stock-related remuneration features are components of total_stock_value
#This was verified using values in the TOTAL key of the dictionary
	#exercised_stock_options
	#restricted_stock
	#restricted_stock_deferred
	#total_stock_value

#Communication features
	#from_messages
	#from_poi_to_this_person
	#from_this_person_to_poi
	#shared_receipt_with_poi
	#to_messages

#All features have numeric values, many of which are missing values (NaN)
#Missing values will be replaced with 0s

print "Count and proportion of 'NaN' for each data element:"
for k in keys:
    nan_count = len([data_dict[k1][k] for k1 in data_dict if data_dict[k1][k] == "NaN"])
    print k, ":", nan_count, " - ", "{:.0%}".format(float(nan_count) / float(record_count))
print "\r\n"

print "Count and proportion of 'NaN' for total remuneration:"
nan_count = 0
for k in data_dict.iterkeys():
    if (data_dict[k]["total_payments"] == "NaN") & (data_dict[k]["total_stock_value"] == "NaN"):
        nan_count += 1
print "total_payments and total_stock_value :", \
    nan_count, " - ", "{:.0%}".format(float(nan_count) / float(record_count))
print "\r\n"

#Over 40% of records are missing values for communication features, possibly because
#24% of records are missing an email_address

#Missing values for many remuneration features are expected, for e.g., director_fees,
#loan_advances, bonus, deferral_payments, exercised_stock_options

#However, only 2% of records are missing both total_payments and total_stock_value,
#which are the aggregate remuneration features

#Due to more complete information in remuneration features, they may be greater factors
#in predicting POI than communication features


#Replacing NaNs with 0s so aggregation, plotting, and classification is possible
#Tried to replace NaNs with median values as certain literature suggests, however,
#classifier performance was far better when replaced with 0s
for k in keys:
    for k1 in data_dict.iterkeys():
        if data_dict[k1][k] == "NaN":
            data_dict[k1][k] = 0.


### Task 1: Remove outliers

#Creating a few lists for outlier detection
total_pmt = [data_dict[k]["total_payments"] for k in data_dict]
total_stock = [data_dict[k]["total_stock_value"] for k in data_dict]

#Checking for outliers on a scatter plot
plt.scatter(total_pmt, total_stock, color = "r")
plt.xlabel("Total Payment")
plt.ylabel("Total Stock")
plt.grid(True)
plt.show()

#The plot shows two outliers
#One of the outlier points is over $100m in total_payments and about $50m in total_stock
#The other outlier point is over $300m in total_payments and over $400m in total_stock
#On browsing the dictionary, the first outlier point is LAY KENNETH L,
#who was the chairman and a real person, and hence belongs in the data set
#The second outlier is not a person but an aggregation of the features - TOTAL,
#which must be removed from the data set
print "OUTLIERS\r\n"
print "LAY KENNETH L:"; pprint.pprint(data_dict["LAY KENNETH L"])
print "\r\n"
print "TOTAL:"; pprint.pprint(data_dict["TOTAL"])
print "\r\n"
del(data_dict["TOTAL"])


### Task 2: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Using SelectKBest for feature selection using the entire data set
#Excluding certain features such as email_address, from_messages, to_messages
#from consideration because logically they cannot be factors in predicting POI
#Tried k = 3 and k = 5, but 3 best features gave performed better
from sklearn.feature_selection import SelectKBest, f_classif
data = featureFormat(data_dict, ["poi", "bonus", "deferral_payments", "deferred_income", \
    "director_fees", "expenses", "loan_advances", "long_term_incentive", "other", \
    "salary", "total_payments", "exercised_stock_options", "restricted_stock", \
    "restricted_stock_deferred", "total_stock_value", "from_poi_to_this_person", \
    "from_this_person_to_poi", "shared_receipt_with_poi"], sort_keys = True)
labels, features = targetFeatureSplit(data)
f_classif_1 = SelectKBest(f_classif, k = 3)
print "FEATURE SELECTION - BEST 3 OF ALL FEATURES\r\n"
print "SelectKBest Indices of 3 Most Important Features:", \
    f_classif_1.fit(features, labels).get_support(indices = True)
print "SelectKBest Scores:", f_classif_1.fit(features, labels).scores_, "\r\n"
#SelectKBest Indices of 3 Most Important Features: [ 0 10 13 ]
#SelectKBest Scores: [ 21.06  0.21  11.59   2.10  6.23  7.24  10.07  4.20  18.57  8.86
#                      25.09  9.34   0.06  24.46  5.34  2.42   8.74 ]
#The above indices and scores indicate that bonus, exercised_stock_options and
#total_stock_value are the 3 most important of all the features tested
#None of the communication related features seem important, but will explore further
features_list = ["poi", "bonus", "exercised_stock_options", "total_stock_value"]

#Exploratory data analysis to confirm optimal features to use with classifiers
#Creating a few lists for exploratory data analysis
poi_bonus = [data_dict[k]["bonus"] for k in data_dict if data_dict[k]["poi"] == True]
npoi_bonus = [data_dict[k]["bonus"] for k in data_dict if data_dict[k]["poi"] == False]

poi_ex_stock = [data_dict[k]["exercised_stock_options"] for k in data_dict if data_dict[k]["poi"] == True]
npoi_ex_stock = [data_dict[k]["exercised_stock_options"] for k in data_dict if data_dict[k]["poi"] == False]

poi_total_stock = [data_dict[k]["total_stock_value"] for k in data_dict if data_dict[k]["poi"] == True]
npoi_total_stock = [data_dict[k]["total_stock_value"] for k in data_dict if data_dict[k]["poi"] == False]

poi_from_poi_msg = [data_dict[k]["from_poi_to_this_person"] for k in data_dict if data_dict[k]["poi"] == True]
npoi_from_poi_msg = [data_dict[k]["from_poi_to_this_person"] for k in data_dict if data_dict[k]["poi"] == False]

poi_to_poi_msg = [data_dict[k]["from_this_person_to_poi"] for k in data_dict if data_dict[k]["poi"] == True]
npoi_to_poi_msg = [data_dict[k]["from_this_person_to_poi"] for k in data_dict if data_dict[k]["poi"] == False]

poi_from_poi_msg_percent = [(float(data_dict[k]["from_poi_to_this_person"]) / float(data_dict[k]["to_messages"]) * 100.) \
    for k in data_dict if (data_dict[k]["poi"] == True) & (data_dict[k]["to_messages"] > 0)]
npoi_from_poi_msg_percent = [(float(data_dict[k]["from_poi_to_this_person"]) / float(data_dict[k]["to_messages"]) * 100.) \
    for k in data_dict if (data_dict[k]["poi"] == False) & (data_dict[k]["to_messages"] > 0)]

poi_to_poi_msg_percent = [(float(data_dict[k]["from_this_person_to_poi"]) / float(data_dict[k]["from_messages"]) * 100.) \
    for k in data_dict if (data_dict[k]["poi"] == True) & (data_dict[k]["from_messages"] > 0)]
npoi_to_poi_msg_percent = [(float(data_dict[k]["from_this_person_to_poi"]) / float(data_dict[k]["from_messages"]) * 100.) \
    for k in data_dict if (data_dict[k]["poi"] == False) & (data_dict[k]["from_messages"] > 0)]

#There are 18 POI and 127 non-POI in the provided data set
print "POI SUMMARY\r\n"
print "No. of POI:", len(poi_bonus)
print "No. of NPOI:", len(npoi_bonus), "\r\n"

#The average and median payout to POI is between 2 to 12 times that to non-POI,
#indicating that POI did benefit far more than non-POI
print "REMUNERATION FEATURES SUMMARY\r\n"
print "Median POI Bonus:", np.median(poi_bonus)
print "Median NPOI Bonus:", np.median(npoi_bonus), "\r\n"

print "Median POI Exercised Stock Options:", np.median(poi_ex_stock)
print "Median NPOI Exercised Stock Options:", np.median(npoi_ex_stock), "\r\n"

print "Median POI Stock:", np.median(poi_total_stock)
print "Median NPOI Stock:", np.median(npoi_total_stock), "\r\n"

print "Avg. POI Bonus:", np.mean(poi_bonus)
print "Avg. NPOI Bonus:", np.mean(npoi_bonus), "\r\n"

print "Avg. POI Exercised Stock Options:", np.mean(poi_ex_stock)
print "Avg. NPOI Exercised Stock Options:", np.mean(npoi_ex_stock), "\r\n"

print "Avg. POI Stock:", np.mean(poi_total_stock)
print "Avg. NPOI Stock:", np.mean(npoi_total_stock), "\r\n"

#Average and median number of messages from/to POI by subject is greater for POI than non-POI
print "COMMUNICATION FEATURES SUMMARY\r\n"
print "POI - Avg. Messages From POI:", np.mean(poi_from_poi_msg)
print "NPOI - Avg. Messages From POI:", np.mean(npoi_from_poi_msg), "\r\n"

print "POI - Avg. Messages To POI:", np.mean(poi_to_poi_msg)
print "NPOI - Avg. Messages To POI:", np.mean(npoi_to_poi_msg), "\r\n"

print "POI - Median Messages From POI:", np.median(poi_from_poi_msg)
print "NPOI - Median Messages From POI:", np.median(npoi_from_poi_msg), "\r\n"

print "POI - Median Messages To POI:", np.median(poi_to_poi_msg)
print "NPOI - Median Messages To POI:", np.median(npoi_to_poi_msg), "\r\n"

#However, when seen as a percentage of total messages sent/received by subject,
#messages from POI to subject do not seem important,
#while messages sent by subject to POI do seem to be a factor
print "POI - Median Messages From POI (% of Total):", \
    np.median(poi_from_poi_msg_percent)
print "NPOI - Median Messages From POI (% of Total):", \
    np.median(npoi_from_poi_msg_percent), "\r\n"

#For at least half the POI, more than a quarter of all messages sent were to other POI
print "POI - Median Messages To POI (% of Total):", \
    np.median(poi_to_poi_msg_percent)
print "NPOI - Median Messages To POI (% of Total):", \
    np.median(npoi_to_poi_msg_percent), "\r\n"

#This first plot indicates that POI exercised stock options and were awarded total stock
#to a greater degree than non-POI, making them both viable factors for predicting POI
plt.scatter(poi_ex_stock, poi_total_stock, color = "r", label = "POI")
plt.scatter(npoi_ex_stock, npoi_total_stock, color = "b", label = "Not POI")
plt.legend()
plt.xlabel("Exercised Stock Options")
plt.ylabel("Total Stock")
plt.grid(True)
plt.show()

#The following plot indicates that bonus could be a factor because barring a few all POI
#received a bonus, while many non-POI did not receive any bonus
plt.scatter(poi_bonus, poi_total_stock, color = "r", label = "POI")
plt.scatter(npoi_bonus, npoi_total_stock, color = "b", label = "Not POI")
plt.legend()
plt.xlabel("Bonus")
plt.ylabel("Total Stock")
plt.yscale("log", nonposy="clip")
plt.grid(True)
plt.show()

#This plot does not indicate a clear trend, the median measures give a better picture
plt.scatter(poi_from_poi_msg, poi_to_poi_msg, color = "r", label = "POI")
plt.scatter(npoi_from_poi_msg, npoi_to_poi_msg, color = "b", label = "Not POI")
plt.legend()
plt.xlabel("From POI Messages")
plt.ylabel("To POI Messages")
plt.grid(True)
plt.show()

#This plot indicates that over 20% of messages sent by almost all POI were to other POI,
#confirming what the median measure above indicated
#So from_this_person_to_poi_percent could be a possible feature to use with classifiers
plt.scatter(poi_from_poi_msg_percent, poi_to_poi_msg_percent, \
    color = "r", label = "POI")
plt.scatter(npoi_from_poi_msg_percent, npoi_to_poi_msg_percent, \
    color = "b", label = "Not POI")
plt.legend()
plt.xlabel("% From POI Messages")
plt.ylabel("% To POI Messages")
plt.grid(True)
plt.show()


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

for k, v in data_dict.iteritems():
    if data_dict[k]["to_messages"] > 0:
        data_dict[k]["from_this_person_to_poi_percent"] = \
            float(data_dict[k]["from_poi_to_this_person"]) / float(data_dict[k]["to_messages"])
    else:
        data_dict[k]["from_this_person_to_poi_percent"] = 0.

my_dataset = data_dict

#Let's test if the newly added feature could replace one of the 3 selected features
data = featureFormat(data_dict, ["poi", "bonus", "exercised_stock_options", \
    "total_stock_value", "from_this_person_to_poi_percent"], sort_keys = True)
labels, features = targetFeatureSplit(data)
f_classif_2 = SelectKBest(f_classif, k = 3)
print "FEATURE SELECTION - BEST 3 OF 4 FEATURES - INCLUDING 1 NEWLY CREATED FEATURE\r\n"
print "SelectKBest Indices of 3 Most Important Features:", \
    f_classif_2.fit(features, labels).get_support(indices = True)
print "SelectKBest Scores:", f_classif_2.fit(features, labels).scores_, "\r\n"
#SelectKBest Indices of 3 Most Important Features: [ 0 1 2 ]
#SelectKBest Scores: [ 17.32  21.15  20.49  2.09 ]

#The above indices and scores indicate that bonus, exercised_stock_options and
#total_stock_value are still the 3 most important features
#Communication-related features do not seem to be important, remuneration seems to be key


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#Either DecisionTree or RandomForest classifiers can be used to achieve better than
#.3 precision and recall
#AdaBoost missed the benchmark for precision and recall

#If casting a wider net is the goal then DecisionTree may be the better option due to its
#generally better recall rate (observed over multiple runs)

#Due to its much better precision rate, RandomForest may be the way to go to ensure
#effort is not wasted investigating people who may not be POI

#AdaBoost
print "ADABOOST CLASSIFIER\r\n"
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
classifier(clf, my_dataset, features_list)
print "AdaBoostClassifier Feature Importances:", clf.feature_importances_, "\r\n"
#Accuracy: 0.78 	Precision: 0.25 	Recall: 0.28

#DecisionTree
print "DECISIONTREE CLASSIFIER\r\n"
from sklearn import tree
clf = tree.DecisionTreeClassifier()
classifier(clf, my_dataset, features_list)
print "DecisionTreeClassifier Feature Importances:", clf.feature_importances_, "\r\n"
#Accuracy: 0.81 	Precision: 0.35 	Recall: 0.39

#RandomForest
print "RANDOMFOREST CLASSIFIER\r\n"
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 5)
classifier(clf, my_dataset, features_list)
print "RandomForestClassifier Feature Importances:", clf.feature_importances_, "\r\n"
#Accuracy: 0.88 	Precision: 0.64 	Recall: 0.39

#Tried each of the above classifiers with their respective 2 best features as scored by
#feature importances, but performance degraded for each classifier

#Did not use SVM classifier as I kept getting a divide by zero error, even after replacing
#0's with a small positive non-zero value
#As a consequence, did not use feature scaling as other classifiers cannot take advantage


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.

#Tuning RandomForestClassifier to get the best n_estimators parameter
#This will determine how many decision trees to create in the forest
#The optimal n_estimators parameter turned out to be 5 or 10, varying with each run
print "RANDOMFOREST CLASSIFIER WITH GRIDCV PARAMETER TUNING\r\n"
features_list = ["poi", "bonus", "exercised_stock_options", "total_stock_value"]
from sklearn.grid_search import GridSearchCV
clf = RandomForestClassifier()
clf = GridSearchCV(clf, param_grid = {"n_estimators": [5, 10, 15]})
classifier(clf, my_dataset, features_list)
print "Best Parameter:", clf.best_params_, "\r\n"
#Accuracy: 0.88 	Precision: 0.62 	Recall: 0.44
#Best Parameter: {'n_estimators': 10}

#Using GridCV for tuning RandomForest parameters improved recall by 5% points,
#while precision degraded by 2% points

#Classifier performance summary
#Note: Precision and recall vary with each run
#ADABOOST CLASSIFIER                    Accuracy: 0.78 	Precision: 0.25 	Recall: 0.28
#DECISIONTREE CLASSIFIER                Accuracy: 0.81 	Precision: 0.35 	Recall: 0.39
#RANDOMFOREST CLASSIFIER                Accuracy: 0.88 	Precision: 0.64 	Recall: 0.39
#RANDOMFOREST CLASSIFIER WITH GRIDCV    Accuracy: 0.88 	Precision: 0.62 	Recall: 0.44


### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)