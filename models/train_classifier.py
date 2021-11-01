import sys
from sqlalchemy import create_engine
import re
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import time
from nltk import download
download('popular')


def load_data(database_filepath):
    '''
    Loads the data from a specified db.

    Input: 
    database_filepath - the path for the database you want to load.

    Output:

    x - A dataframe with the features.
    y - A dataframe with the labels.
    category_names - A list of the categories names.
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messsages', engine)
    x = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    return x, y, category_names


def tokenize(text):
    '''
    tokenizes the text.
    
    Input:
    text - a text string.
    
    Output:
    a list of tokenized words after Lemmatizing.
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Normalize text by removing anything but letters and numbers
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize words in the text
    words = word_tokenize(text)
    # Lemmatize words
    lemmatized = [WordNetLemmatizer().lemmatize(w).strip() for w in words]
    return lemmatized


def build_model():
    '''
    Builds a model with a pipeline.

    Doesn't accept any args.

    Returns a GridSearch object.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__max_depth': [10, 20, 50],
        'clf__estimator__n_estimators': [10, 20, 100],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluates the model.

    Input:
    model - the model you want to be evaluated.
    X_test - A dataframe with the testing features.
    Y_test - A dataframe with the testing labels.
    category_names - A list of the categories names.
    '''
    Y_pred = model.predict(X_test)
    avg_accuracy = []
    for i in range(len(category_names)):
        print('("Category: {}\n{}'.format(category_names[i], classification_report(
            Y_test.iloc[:, i].values, Y_pred[:, i])))

        avg_accuracy.append(accuracy_score(
            Y_test.iloc[:, i].values, Y_pred[:, i]))

    print('Overall model accuracy: {}'.format(
        round(sum(avg_accuracy)/len(avg_accuracy), 2)))


def save_model(model, model_filepath):
    '''Saves the model into the path specified.'''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        start = time.time()
        model.fit(X_train, Y_train)
        print('Training took {} minutes'.format(
            round((time.time() - start)/60, 2)))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
