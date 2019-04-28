import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data_attributes = []
data_labels = []

attributes_train = []
attributes_test = []
labels_train = []
labels_test = []

def process(attributes, labels):
    global data_attributes, data_labels, attributes_train, attributes_test, labels_train, labels_test
    data_attributes = attributes
    data_labels = labels

    attributes_train, attributes_test, labels_train, labels_test = train_test_split(data_attributes, data_labels, test_size=0.2, random_state = 0)
    ScaleData()
    result = train()
    print_report(result)

def ScaleData():
    global attributes_test, attributes_train
    sc = StandardScaler()
    attributes_train = sc.fit_transform(attributes_train)
    attributes_test = sc.fit_transform(attributes_test)

def train():
    global attributes_train, labels_train, attributes_test
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(attributes_train, labels_train)
    label_predict = classifier.predict(attributes_test)
    return label_predict

def print_report(label_predict):
    global labels_test
    #print(confusion_matrix(labels_test, label_predict))
    #print(classification_report(labels_test, label_predict))
    print("Accuracy with Random Forest: " + str(accuracy_score(labels_test, label_predict)))
