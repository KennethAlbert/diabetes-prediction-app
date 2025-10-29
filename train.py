# train_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model():
    # Load the dataset
    df = pd.read_csv('diabetes.csv')
    
    # Separate the data and the labels
    x = df.drop(columns='Outcome', axis=2)
    y = df['Outcome']
    
    # Data standardization
    scaler = StandardScaler()
    scaler.fit(x)
    standardized_data = scaler.transform(x)
    
    x = standardized_data
    y = df['Outcome']
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
    
    # Train classifier
    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)
    
    # Calculate accuracy
    x_train_prediction = classifier.predict(x_train)
    training_data_accuracy = accuracy_score(x_train_prediction, y_train)
    print("Accuracy score on training data:", training_data_accuracy)
    
    x_test_prediction = classifier.predict(x_test)
    test_data_accuracy = accuracy_score(x_test_prediction, y_test)
    print("Accuracy score on testing data:", test_data_accuracy)
    
    # Save the scaler and classifier
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(classifier, 'models/classifier.pkl')
    
    print("Scaler and classifier saved successfully in 'models/' folder.")
    
    return scaler, classifier, test_data_accuracy

if __name__ == "__main__":
    train_and_save_model()