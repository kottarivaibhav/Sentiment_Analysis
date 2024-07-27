import re
import pandas as pd
from textblob import TextBlob
from flask import Flask, render_template, request, redirect, url_for
from io import StringIO

app = Flask(__name__)
app.static_folder = 'static'

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return "positive"
    elif analysis.sentiment.polarity == 0:
        return "neutral"
    else:
        return "negative"

def get_tweets_from_csv(file_content):
    try:
        data = pd.read_csv(StringIO(file_content))
        tweets = data.to_dict(orient='records')
        for tweet in tweets:
            tweet['sentiment'] = get_tweet_sentiment(tweet['content'])
        return tweets
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return []

@app.route('/')
def home():
    return render_template("features.html")

@app.route("/predict", methods=['POST'])
def pred():
    if request.method == 'POST':
        try:
            csv_file = request.files['csv_file']
            file_content = csv_file.read().decode('utf-8')
            fetched_tweets = get_tweets_from_csv(file_content)
            return render_template('result.html', result=fetched_tweets)
        except Exception as e:
            return f"Error: {str(e)}"

@app.route("/predict1", methods=['POST'])
def pred1():
    if request.method == 'POST':
        text = request.form['txt']
        blob = TextBlob(text)
        if blob.sentiment.polarity > 0:
            text_sentiment = "positive"
        elif blob.sentiment.polarity == 0:
            text_sentiment = "neutral"
        else:
            text_sentiment = "negative"
        return render_template('result1.html', msg=text, result=text_sentiment)

#if __name__ == '__main__':
 #   app.run(debug=True)
