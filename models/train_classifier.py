import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['stopwords', 'punkt', 'wordnet'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    '''
    load dataset
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    A tokenization function to process the text data
    ---
    Input - text
    output - cleaned tokens
    '''
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    token = word_tokenize(text)
    
    clean = []
    for tok in token:
        tok_clean = lemmatizer.lemmatize(tok).lower().strip()
        if tok_clean not in stop_words:
            clean.append(tok_clean)
    return clean


def build_model():
    '''
    Build pipeline and use grid search
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    
    parameters = {
    'vect__max_df': (0.5, 0.75, 1.0)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluate model and print f1 score, precision, and recall'''
    Y_pred = model.predict(X_test)
    for i in range(Y_pred.shape[1]):
        print(classification_report(np.array(Y_test)[:,i], Y_pred[:,i], target_names=category_names))


def save_model(model, model_filepath):
    '''Save model as pkl file'''
    pkl_filename = "classifier.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(cv, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        print(database_filepath)
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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