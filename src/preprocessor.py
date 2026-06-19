import pandas as pd

def preprocess_data(X):

    # As Total Charges is already been handled we will just preprocess these 2 columns before passing to the pipeline
    
    X['SeniorCitizen'] = pd.to_numeric(X['SeniorCitizen'],errors='raise')
    X['gender'] = X['gender'].map({'Male':1,"Female":0})

    return X
