import os
import re
import pandas as pd
from textblob import TextBlob
from flask import Flask, render_template, request, redirect, url_for, send_file
from io import StringIO
import tempfile
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

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

def preprocess_text(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)
    stop_words = set(stopwords.words('english'))
    tweet = ' '.join([word for word in tweet.split() if word not in stop_words])
    return tweet

def get_polarity_words(tweet):
    analysis = TextBlob(tweet)
    words = analysis.words
    word_polarities = {word: TextBlob(word).sentiment.polarity for word in words}
    sorted_words = sorted(word_polarities.items(), key=lambda item: item[1], reverse=True)
    top_words = [word for word, polarity in sorted_words[:5]]  # Get top 5 words by polarity
    return ', '.join(top_words), analysis.sentiment.polarity

def get_tweets_from_csv(file_content):
    try:
        data = pd.read_csv(StringIO(file_content))
        tweets = data.to_dict(orient='records')
        for tweet in tweets:
            tweet['sentiment'] = get_tweet_sentiment(tweet['content'])
            tweet['preprocessed_content'] = preprocess_text(tweet['content'])
            top_words, polarity_score = get_polarity_words(clean_tweet(tweet['content']))
            tweet['top_polarity_words'] = top_words
            tweet['polarity_score'] = polarity_score
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
            
            processed_df = pd.DataFrame(fetched_tweets)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            processed_df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name
            temp_file.close()
            
            return render_template('result.html', result=fetched_tweets, csv_download=True, temp_file_path=temp_file_path)
        except Exception as e:
            return f"Error: {str(e)}"

@app.route('/download_csv')
def download_csv():
    temp_file_path = request.args.get('temp_file_path')
    if temp_file_path and os.path.exists(temp_file_path):
        return send_file(temp_file_path, mimetype='text/csv', download_name='processed_tweets.csv', as_attachment=True)
    else:
        return "File not found", 404

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

if __name__ == '__main__':
    app.run(debug=True)
