import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xlsxwriter
from collections import Counter

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath).dropna(axis=1)
    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])
    return data, encoder

def split_data(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:, -1]
    return X, y

def train_models(X_train, y_train):

    svm_model = SVC(probability=True)
    nb_model = GaussianNB()
    rf_model = RandomForestClassifier(random_state=18)
    svm_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    
    models = {
        "SVM": svm_model,
        "Naive Bayes": nb_model,
        "Random Forest": rf_model
    }
    
    return models

def evaluate_models(models, X_test, y_test):
    
    results = {
        "Model":[],
        "Accuracy":[],
        "AUC":[]
        }
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        results["Model"].append(name)
        results["Accuracy"].append(acc)
        results["AUC"].append(roc_auc)
    return pd.DataFrame(results)

def encodeInput(X_train, encoder):
    symptoms = X_train.columns.values

    # Creating a symptom index dictionary to encode the
    # input symptoms into numerical form
    symptom_index = {}

    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index

    data_dict = {
        "symptom_index":symptom_index,
        "predictions_classes":encoder.classes_
    }
    
    return data_dict

def predictDisease(filepath, models, data_dict):
    
    data = pd.read_csv(filepath).dropna(axis=1)
    
    for index, rows in data.iterrows():
        symptoms = rows["Symptoms"]
        # predictions ={keys + " Prediction":[] for keys in models.keys()}
        predictions_list = []
        
        symptoms = symptoms.split(",")
        
        # creating input data for the models
        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms:
            sym_index = data_dict["symptom_index"][symptom]
            input_data[sym_index] = 1
            
        # reshaping the input data and converting it
        # into suitable format for model predictions
        input_data = np.array(input_data).reshape(1,-1)
        
        for name, model in models.items():
            
            y_pred = data_dict["predictions_classes"][model.predict(input_data)[0]]
            predictions_list.append(y_pred)
            data.loc[index, name + " Prediction"] = y_pred
        
        final_prediction = Counter(predictions_list).most_common(1)[0][0]
        data.loc[index, "Final Prediction"] = final_prediction
    
    return data

def main():
    train_data, encoder = load_and_preprocess_data('../dataset/Training.csv')
    test_data, encoder = load_and_preprocess_data('../dataset/Testing.csv')
    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    results.to_excel('../output/model_results.xlsx', index=False)
    
    data_dict = encodeInput(X_train, encoder)
    output = predictDisease('../dataset/Symptoms.csv', models, data_dict)
    output.to_excel('../output/predictions.xlsx', index=False)

if __name__ == "__main__":
    main()
