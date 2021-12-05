import json
import plotly
import pandas as pd
import joblib
from sqlalchemy import create_engine

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Line


app = Flask(__name__)


def stacked_bar_data():
    y = df.iloc[:, 4:]
    cnt_1 = []
    cnt_0 = []
    row_cnt = y.shape[0]

    for column in y.columns:
        count_of_ones = y[column].sum()
        cnt_1.append(count_of_ones)
        count_of_zeros = row_cnt - count_of_ones
        cnt_0.append(count_of_zeros)

    df_target = pd.DataFrame({'1': cnt_1, '0': cnt_0})
    df_target.index = y.columns
    df_target = df_target.reset_index().rename(columns = {'index': 'target_category'})
    df_target = df_target.sort_values(by = '1', ascending=False)
    return df_target


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
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


# load data
engine = create_engine('sqlite:///../UdacityDisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../nlp_multi_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # bar graph data
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # line graph data
    category_counts = stacked_bar_data()

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Line(
                    x=category_counts["target_category"],
                    y=category_counts["1"],
                    line=dict(color="orange"),
                    mode = "lines+markers"
                )
            ],
            'layout': {
                'title': 'Number of 1s Per Target Category',
                'yaxis': {
                    'title': 'Count of Class 1s'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
