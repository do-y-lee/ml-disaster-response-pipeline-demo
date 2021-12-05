import re
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

import nltk
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)
    # remove rows with 2's under related column before split into X and y
    df = df[df.related != 2]
    # split data
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns.values
    return X, y, category_names


def tokenize(text):
    """
    Input text is normalized and then tokenized. Next, the tokenized text is lemmatized and outputs a list of tokens.

        Parameters:
            text (str): Cleaned text string is passed into the function as input.

        Returns:
            clean_tokens (list): Generates a list of cleaned tokens.
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # identify urls
    detected_urls = re.findall(url_regex, text)
    # replace urls
    for detected_url in detected_urls:
        text = text.replace(detected_url, "urlplaceholder")

    # extract word tokens from provided text
    tokens = nltk.word_tokenize(text)
    # lemmanitize by removing inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()
    # list of cleaned tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    # extract the starting verb of a sentence (new feature)
    @staticmethod
    def starting_verb(text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(tokenize(sentence))
            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_train_model(X_train, y_train):
    """
    Function is wrapped around Sklearn's Pipeline object. Pipeline object is designed to stack various data
    transformation and model training layers to output a trained classifier. Sklearn's GridSearchCV is
    used to tune the hyper-parameters to further tune the model during training.

        Parameters:
            X_train (pd.DataFrame): train data
            y_train (pd.DataFrame): train multiple target variables

        Returns:
            model (object): Fitted and tuned multioutput classifier pipeline object
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer = tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__count_vectorizer__max_df': (0.5, 0.75, 1.0),
        'clf__estimator__min_samples_split': [2, 3, 4],
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=5)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test.values, y_pred_test, target_names=category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, y, category_names = load_data(database_filepath)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        print('Training model...')
        model = build_train_model(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
