import sys
import pandas as pd
import numpy as np
import re
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Funtion to read df from sqlite

    INPUT
    database_filepath - path ot the database filename

    OUTPUT
    X - Feature for training  
    Y - Feature to be evaluated
    category_names - Array of category names.
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM Disaster_Response_Pipeline",engine)
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X,y,category_names

url_regex= 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F]))+'
def tokenize(text):
    
    """
    Funtion to tokenize text

    INPUT
    text - input text

    OUTPUT
    clean_tokens - cleaned tokenized text.
    """
    detected_urls=re.findall(url_regex,text)
    for url in detected_urls:
        text=text.replace(url,"urlplaceholder")
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok=lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Funtion to build model pipeline 

    INPUT
    None

    OUTPUT
    model - model built from Gridsearch using pipeline and params. 
    """
    # Build a ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators':[30,50],'clf__estimator__min_samples_split':[3,4]}
    model = GridSearchCV(pipeline,param_grid=parameters)
    return model
def evaluate_model(model, X_test, y_test, category_names):
    """
    Funtion to evaluate model performance

    INPUT
    
    model - model to train on the dataset
    X_test - test data for training
    y_test - test data for evaluating
    category_names - Array of category names

    OUTPUT
    
    Print accuracy score and classification report
    
    """
    y_pred=model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    report = classification_report(y_test, y_pred, target_names=category_names)
    print(accuracy)
    print('\n',report)
    


def save_model(model, model_filepath):
    """
    Funtion to save model as pkl file

    INPUT
    
    model - model to save
    model_filepath - Path to save the trained model

    OUTPUT
    None
    """
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """ Function to load data,build,train,test and evaluate model ."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()