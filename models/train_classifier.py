import sys
import pandas as pd
import re
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pickle
from sklearn.svm import LinearSVC



def load_data(database_filepath):

    """
    INPUT: filepath of database to be read
    OUTPUT: X : dataframe with feature variables, 
            Y : dataframe with target variables,
            category_names: list of categories to be predicted
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    X = df['message']
    Y = df.drop(columns=['message', 'id','original', 'genre'])
    category_names=Y.columns

    return X, Y, category_names


def tokenize(text):

    """
    The function normalizes, tokenizes, removes stop words and
    lemmatizes the given text

    INPUT : text to be tokenized
    OUTPUT: list of tokenized words
    """
    #normalize

    text = re.sub(r"[^a-zA-Z0-9]", " ",text.lower())

    #tokenize
    token_words=word_tokenize(text)

    #remove stopwords from the token words
    Stopwords=stopwords.words('english')
    token_words = [w for w in token_words if w not in Stopwords]

    #lemmatize each token word
    token_words = [WordNetLemmatizer().lemmatize(w) for w in token_words]
    
    return token_words


def build_model():
    """
    The function builds a model for classifying the messages
    INPUT: None
    OUTPUT: Trained model
    """
    #create pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfdif',TfidfTransformer()),
        ('mlo',MultiOutputClassifier(AdaBoostClassifier()))
        ])

    #list the gridsearch parameters
    parameters = {'vect__ngram_range': ((1, 1),(1, 2)),
             'mlo__estimator__n_estimators': [50, 60],
             }

    #model with best parameters from gridsearch
    cv = GridSearchCV(pipeline, parameters)
    return cv




def evaluate_model(model, X_test, Y_test, category_names):

    """
    The function shows the performance of a given model on give test set
    INPUT: model: the model performance to be evaluated
           X : test set dataframe with feature variables, 
           Y : test set dataframe with target variables,
           category_names: list of categories to be predicted
    OUTPUT: Prints the  f1 score, precision and recall for the test set each category
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == Y_pred)))


def save_model(model, model_filepath):

    """
    The function saves the given model as pickle file using the given filepath
    INPUT: Trained model, filepath to save the model
    OUTPUT: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
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