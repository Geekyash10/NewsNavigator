from flask import Flask, request, jsonify
from flask_cors import CORS
import regex
import string
import pandas as pd
from nltk import word_tokenize , sent_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import LabelEncoder
import joblib
import re


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
import requests
from bs4 import BeautifulSoup

stop_words=set(stopwords.words('english'))
punc=string.punctuation

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

abbreviation_dict = {
    'LOL': 'laugh out loud',
    'BRB': 'be right back',
    'OMG': 'oh my god',
    'AFAIK': 'as far as I know',
    'AFK': 'away from keyboard',
    'ASAP': 'as soon as possible',
    'ATK': 'at the keyboard',
    'ATM': 'at the moment',
    'A3': 'anytime, anywhere, anyplace',
    'BAK': 'back at keyboard',
    'BBL': 'be back later',
    'BBS': 'be back soon',
    'BFN': 'bye for now',
    'B4N': 'bye for now',
    'BRB': 'be right back',
    'BRT': 'be right there',
    'BTW': 'by the way',
    'B4': 'before',
    'B4N': 'bye for now',
    'CU': 'see you',
    'CUL8R': 'see you later',
    'CYA': 'see you',
    'FAQ': 'frequently asked questions',
    'FC': 'fingers crossed',
    'FWIW': 'for what it\'s worth',
    'FYI': 'For Your Information',
    'GAL': 'get a life',
    'GG': 'good game',
    'GN': 'good night',
    'GMTA': 'great minds think alike',
    'GR8': 'great!',
    'G9': 'genius',
    'IC': 'i see',
    'ICQ': 'i seek you',
    'ILU': 'i love you',
    'IMHO': 'in my honest/humble opinion',
    'IMO': 'in my opinion',
    'IOW': 'in other words',
    'IRL': 'in real life',
    'KISS': 'keep it simple, stupid',
    'LDR': 'long distance relationship',
    'LMAO': 'laugh my a.. off',
    'LOL': 'laughing out loud',
    'LTNS': 'long time no see',
    'L8R': 'later',
    'MTE': 'my thoughts exactly',
    'M8': 'mate',
    'NRN': 'no reply necessary',
    'OIC': 'oh i see',
    'PITA': 'pain in the a..',
    'PRT': 'party',
    'PRW': 'parents are watching',
    'QPSA?': 'que pasa?',
    'ROFL': 'rolling on the floor laughing',
    'ROFLOL': 'rolling on the floor laughing out loud',
    'ROTFLMAO': 'rolling on the floor laughing my a.. off',
    'SK8': 'skate',
    'STATS': 'your sex and age',
    'ASL': 'age, sex, location',
    'THX': 'thank you',
    'TTFN': 'ta-ta for now!',
    'TTYL': 'talk to you later',
    'U': 'you',
    'U2': 'you too',
    'U4E': 'yours for ever',
    'WB': 'welcome back',
    'WTF': 'what the f...',
    'WTG': 'way to go!',
    'WUF': 'where are you from?',
    'W8': 'wait...',
    '7K': 'sick laughter',
    'TFW': 'that feeling when',
    'MFW': 'my face when',
    'MRW': 'my reaction when',
    'IFYP': 'i feel your pain',
    'LOL': 'laughing out loud',
    'TNTL': 'trying not to laugh',
    'JK': 'just kidding',
    'IDC': 'i don’t care',
    'ILY': 'i love you',
    'IMU': 'i miss you',
    'ADIH': 'another day in hell',
    'IDC': 'i don’t care',
    'ZZZ': 'sleeping, bored, tired',
    'WYWH': 'wish you were here',
    'TIME': 'tears in my eyes',
    'BAE': 'before anyone else',
    'FIMH': 'forever in my heart',
    'BSAAW': 'big smile and a wink',
    'BWL': 'bursting with laughter',
    'LMAO': 'laughing my a** off',
    'BFF': 'best friends forever',
    'CSL': 'can’t stop laughing',
}

y = pd.read_csv('trans_data.csv')

model = joblib.load('modelCate.pkl')
modelFake = joblib.load('modelFake.pkl')

le = LabelEncoder()
le.fit(y['category_grouped'])

def scrape_article_content(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find the article body
        article_body = soup.find('article')
        if not article_body:
            # Fallback to <div> with specific class names
            article_body = soup.find('div', class_='main-content')

        # If both `article` and `div` tags are missing, concatenate all <p> tags
        if not article_body:
            article_content = " ".join(p.get_text() for p in soup.find_all('p'))
        else:
            # Extract text content from the article body
            paragraphs = article_body.find_all('p')
            article_content = ' '.join([para.get_text() for para in paragraphs])

        # Return cleaned-up article content if found
        return article_content.strip() if article_content else None

    except Exception as e:
        print(f"Error while scraping: {e}")
        return None
    


def preprocessText(text):
    """
    Perform text preprocessing aligned with wordpre while adding tokenization, stopwords, and abbreviation handling.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove text inside brackets
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove special characters and non-alphanumeric content
    text = re.sub("\\W", " ", text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)
    
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    # Remove newlines
    text = re.sub('\n', '', text)
    
    # Remove words containing numbers
    text = re.sub(r'\w*\d\w*', '', text)
    
    # Remove emojis
    emoji_pattern = regex.compile(r'\p{Emoji}', flags=regex.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove stopwords
    words = text.split()
    text = " ".join(word for word in words if word not in stop_words)
    
    # Replace abbreviations
    for abbreviation, full_form in abbreviation_dict.items():
        text = text.replace(abbreviation, full_form)
    
    # Tokenize text
    words_list = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    text = ' '.join(' '.join(words) for words in words_list)
    
    return text
    

@app.route('/api/scrape', methods=['POST'])
def scrape():
    try:
        data = request.json
        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Scrape and preprocess article content
        article_content = scrape_article_content(url)
        if not article_content:
            return jsonify({'error': 'Unable to scrape the article'}), 500

        article_content = preprocessText(article_content)
        article_content = [article_content]

        # Predict category and fake/real classification
        predicted_value = model.predict(article_content)
        predicted_category = le.inverse_transform(predicted_value)[0]
        fake_news_pred = modelFake.predict(article_content)[0]

        fake_status = "Fake" if fake_news_pred == 0 else "Real"

        # Send results
        return jsonify({
            'category': predicted_category,
            'fake_status': fake_status
        }), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/description', methods=['POST'])
def description():
    try:
        data = request.json
        description = data.get('description')
        if not description:
            return jsonify({'error': 'Description is required'}), 400

        # Preprocess and predict
        processed_description = preprocessText(description)
        processed_description = [processed_description]

        predicted_value = model.predict(processed_description)
        predicted_category = le.inverse_transform(predicted_value)[0]
        fake_news_pred = modelFake.predict(processed_description)[0]

        fake_status = "Fake" if fake_news_pred == 0 else "Real"

        # Send results
        return jsonify({
            'category': predicted_category,
            'fake_status': fake_status
        }), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'Failed to process description'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=3000)
