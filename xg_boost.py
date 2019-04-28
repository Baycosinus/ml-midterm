import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

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
    attributes_train, attributes_test, labels_train, labels_test = train_test_split(data_attributes, data_labels, test_size=0.2, random_state=0)

    model = XGBClassifier()
    model.fit(attributes_train, labels_train)

    pred = model.predict(attributes_test)
    predictions = [round(value) for value in pred]

    accuracy = accuracy_score(labels_test,predictions)
    print("Accuracy with xgBoost: %.2f" % (accuracy * 100.0))
