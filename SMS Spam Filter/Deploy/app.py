import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from flask import Flask, request, jsonify, render_template
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

app = Flask(__name__,template_folder='templates')
model = pickle.load(open('best_model.pkl', 'rb'))
vec = pickle.load(open('best_vec.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
	# Import nltk resources

    resources = ["wordnet", "stopwords", "punkt", "averaged_perceptron_tagger", "maxent_treebank_pos_tagger"]
    for resource in resources:
        try:
            nltk.data.find('tokenizers/' + resource)
        except LookupError:
            nltk.download(resource)


    # Create stopwords list        
    STOPWORDS = set(stopwords.words('english'))

    # Define main text cleaning function
    def clean_text(text):
        """
        Return a processed version of the text given
        """
        # Turn all text into lower case
        text = text.lower()
        
        
        # Remove stopwords
        text = ' '.join(word for word in text.split() if word not in STOPWORDS)
        
        # Remove all punctuations
        #punctuations = '''!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'''
        text = ' '.join(word.strip(string.punctuation) for word in text.split())
        
        # Remove numerics
        text = re.sub(r'\d+', '', text)
        
        # Removing Extra spaces if any
        text = re.sub(r'[\s]+', ' ', text)
    
        # Convert 
        return text
    from sklearn.base import BaseEstimator, TransformerMixin

    class MetaDataExtractor(BaseEstimator, TransformerMixin):
        """
        Takes in text series, outputs meta-features
        """

        def __init__(self):
            pass

        def extract_meta_data(self, message):
            """
            This function preprocess one text message, creating a list of metadata.
            The message argument is a message.
            """

            # Replace email addresses with 'EmAd'
            message = re.sub(r'[^\s]+@.[^\s]+', '{EmAd}', message)

            # Replace URLs with 'Url'
            message = re.sub(r'http[^\s]+', '{Url}', message)

            # Replace money symbols with 'MoSy'
            message = re.sub(r'£|\$', '{MoSy}', message)

            # Replace 10 or 11 digit phone numbers
            message = re.sub(r'0?(\d{10,}?)','{PhNu}', message)

            # Derive tokens
            token = nltk.word_tokenize(message)

            # Derive number of tokens
            n_token = len(token)

            # Derive the average length of a token
            avg_len = np.mean([len(word) for word in message.split()])

            # Derive the number of numerics
            n_num = len([tok for tok in message if tok.isdigit() or tok == '{PhNu}'])

            # Derive if the message has numerics
            has_num = np.where(n_num > 0,1,0)

            # Derive the number of uppercased words
            n_uppers = len([word for word in message if word.isupper()])

            # Derive the number of English stop words
            n_stops = len([word for word in message if word in stopwords.words('english')])

            # Derive the symbol columns
            has_email = np.where('{EmAd}' in message,1, 0)
            has_money = np.where('{MoSy}' in message,1, 0)
            has_phone = np.where('{PhNu}' in message,1,0)
            has_url = np.where('{Url}' in message,1,0)

            return np.array([n_token, avg_len, n_num, has_num, n_uppers, n_stops, has_email, has_money, has_phone, has_url])
        
        def transform(self, message, y=None):
            """
            Tranform the meta-data features extracted and convert into dataframe format
            """
            return self.extract_meta_data(message)

        def fit(self, message, y=None):
            """Returns `self` unless something different happens in train and test"""
            return self
    message = request.form['message']
    message_processed = clean_text(message)
    sparse_feat = vec.transform([message_processed])
    dense_feat = MetaDataExtractor().fit_transform(message)
    dense_feat.reshape(-1, 1).reshape(1, -1)
    dense_feat = coo_matrix(MinMaxScaler().fit_transform(dense_feat.reshape(-1, 1)))
    final_feat = hstack([sparse_feat, dense_feat.reshape(1, -1).astype(float)]) 
    prediction = np.where(model.predict(final_feat) == [1], "SPAM", "NORMAL")
    return render_template('index.html', prediction_text ="""Spam-Bot-9000 says the sms: "{}"is DEFINITELY a {}""".format(message, prediction))


if __name__ == '__main__':
	app.run(host='0.0.0.0')
