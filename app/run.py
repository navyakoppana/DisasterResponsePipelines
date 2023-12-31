import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Funtion to tokenize text

    INPUT
    text - input text

    OUTPUT
    clean_tokens - cleaned tokenized text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster_Response_Pipeline', engine)

# load model
model = joblib.load("../models/classifier.pkl")
def return_graphs():
    """ Creates two plotly visualizations"""
    # Number of Messages per Category
    graph_one = [] 
    message_counts = df.drop(['id','message','original','genre'], axis=1).sum().sort_values()
    message_names = list(message_counts.index)
    graph_one.append(
        Bar(
        y = message_names,
        x = message_counts, 
        orientation='h',
      )
    )
    
    layout_one = dict(title = 'Number of Messages per Category',
                xaxis = dict(title = "Count"),
                yaxis = dict(title = "Message_names",tickfont=dict(size=8),tickangle=-30),
                )

    graph_two = [] 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_two.append(
      Bar(
        x = genre_names,
        y = genre_counts,
      )
    )

    layout_two = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = "Genre",),
                yaxis = dict(title = "Count"),
                )


    # append all charts to the graphs list
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    
    
    return graphs

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    graphs = return_graphs()
    
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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()