#%%
!wget "https://www.dropbox.com/scl/fi/yurr7e4jpjs97hwc7awam/english_news_dataset.csv?rlkey=s2e4ks6787ufnckz65iiqi2u8&st=8i9ah0ni&dl=0"
# !mv "/content/english_news_dataset.csv?rlkey=s2e4ks6787ufnckz65iiqi2u8&st=8i9ah0ni&dl=0" data.csv
!ls
#%%
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import regex
import re
from nltk import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
#%%

#%%
df=pd.read_csv('./data.csv')
df.shape
#%%
df.head()
#%%
from sklearn.model_selection import StratifiedKFold, cross_val_score
threshold = 5

# Identify classes with fewer instances
class_counts = df['News Categories'].value_counts()
rare_classes = class_counts[class_counts < threshold].index

# Group rare classes into a broader category 'Other'
df['category_grouped'] = df['News Categories'].apply(lambda x: 'Other' if x in rare_classes else x)
#%%
df["News Categories"]
#%%
df['category_grouped']
#%%
df.info()
#%%
df.isnull().sum()
#%%
df.duplicated().sum()
#%%
df["Date"].head()
#%%
df["News Categories"].unique().sum()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
top_n = 5
top_categories = df['News Categories'].value_counts().nlargest(top_n).index

df_top = df[df['News Categories'].isin(top_categories)]

sns.countplot(x='News Categories', data=df_top, palette='viridis')
plt.title(f'Top {top_n} News Categories')
plt.xlabel('Categories')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show();
#%%
import string
string.punctuation
punc=string.punctuation
#%%
def remove_punc(text):
    return text.translate(str.maketrans('', '',punc))

df["News Categories"]=df["News Categories"].apply(remove_punc)
df.head()
#%%
df['Date'] = pd.to_datetime(df['Date'],format='mixed',dayfirst=True)

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

df.head()
#%%
df=df.drop('Date',axis=1)
#%%
df.head()
#%%
df["Content"]=df["Content"].str.lower()
df.head()
#%%
# !pip install beautifulsoup4
#%%
from bs4 import BeautifulSoup

## check if there is html tags

def has_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return bool(soup.find())

df['has_html_tags'] = df['Content'].apply(has_html_tags)
#%%
count_true = df['has_html_tags'].sum()
count_true
#%%
df = df.drop('has_html_tags', axis=1)
df
#%%
# !pip install emoji
# !pip install --upgrade emoji
# !pip install regex
#%%
import regex
def has_emoji(text):
    emoji_pattern = regex.compile(r'\p{Emoji}', flags=regex.UNICODE)
    return bool(emoji_pattern.search(text))


has_emojis =  df['Content'].apply(has_emoji)

has_emojis
#%%
has_emojis.sum()
#%%
def remove_emojis(text):
    emoji_pattern = regex.compile(r'\p{Emoji}', flags=regex.UNICODE)
    return emoji_pattern.sub('', text)

df['Content'] = df['Content'].apply(remove_emojis)
#%%
has_emojis =  df['Content'].apply(has_emoji)
has_emojis
#%%
has_emojis.sum()
#%%
import re
def remove_url(text):
    pattern=re.compile(r'https?://\S+|www\.S+')
    return pattern.sub(r'',text)
df["Content"]=df["Content"].apply(remove_url)
#%%
import string
string.punctuation

punc=string.punctuation
#%%
def remove_punc(text):
    return text.translate(str.maketrans('', '',punc))
df["Content"]=df["Content"].apply(remove_punc)
#%%
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
#%%
stop_words=set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

df["Content"]=df["Content"].apply(lambda x: remove_stopwords(x))
#%%
import nltk
nltk.download('punkt')

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
#%%
def replace_abbreviations(text, abbreviation_dict):
    for abbreviation, full_form in abbreviation_dict.items():
        text = text.replace(abbreviation, full_form)
    return text
df['Content'] = df['Content'].apply(lambda x: replace_abbreviations(x,abbreviation_dict))
#%%
df.head()
#%%
import nltk
nltk.download('punkt')
from nltk import word_tokenize , sent_tokenize

def tokenize_text(text):
    # Tokenize each sentence into words
    words_list = [word_tokenize(sentence) for sentence in sent_tokenize(text)]

    words = ' '.join(' '.join(words) for words in words_list)

    return words

df["Content"] = df["Content"].apply(tokenize_text)

df.head()
#%%
!pip install imbalanced-learn
#%%
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
#%%
X = df['Content']
y = df['category_grouped']

# Encoding labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

class_weights_train = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)

X
#%%
y.shape
#%%
y_encoded
#%%
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
#%%
# Multinomial Naive Bayes with Bag of Words
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,classification_report


model = make_pipeline(CountVectorizer(), MultinomialNB())

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"MultinomialNB with Bag of Words accuracy: {accuracy:.3f}")
# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
#%%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

cv_scores = cross_val_score(model, X, y_encoded, cv=StratifiedKFold(n_splits=3, shuffle=True), scoring='accuracy')

print(f"Cross-Validation Scores:{cv_scores}")

print(f"Mean Accuracy: {np.mean(cv_scores):.2f}")
#%%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_dist = {
    'countvectorizer__max_features': [5000, 10000, None],
    'countvectorizer__ngram_range': [(1, 1), (1, 2)],
    'multinomialnb__alpha': uniform(0.1, 2.0)  # Example range for alpha
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, scoring='accuracy', cv=cv, verbose=1, n_jobs=1)
random_search.fit(X, y_encoded)
#%%
best_params = random_search.best_params_
print("Best Parameters:", best_params)
#%%
best_model = random_search.best_estimator_

best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_best)

print(f"Best Model Accuracy: {accuracy:.3f}")
#%%
# Inverse transform the predicted labels to get the original class labels
predicted_labels_original = le.inverse_transform(y_pred_best)
#%%
correct_predictions = sum(y_test == y_pred_best)
wrong_predictions = len(y_test) - correct_predictions
print(f'Correct Predictions: {correct_predictions}, Wrong Predictions: {wrong_predictions}')
#%%
labels = ['Correct Predictions', 'Wrong Predictions']
values = [correct_predictions, wrong_predictions]

plt.bar(labels, values, color=['green', 'red'])
plt.title('Correct vs Wrong Predictions')
plt.xlabel('Prediction Outcome')
plt.ylabel('Number of Samples')
plt.show()
#%%
#final dataframe with text and predicted labels
final_df = pd.DataFrame({'Content': X_test, 'Predicted_Labels': predicted_labels_original, 'Actual_Labels': le.inverse_transform(y_test)})

final_df.head()
#%%
final_df.to_csv('finalDF.csv')
#%%
import joblib

# Save the trained pipeline model
# joblib.dump(model, 'model.pkl')
#%%
# random_search.save('random_search.h5')
joblib.dump(random_search, 'random_search.pkl')
#%%
# best_model.save('best_model.h5')
joblib.dump(best_model, 'best_model.pkl')
#%%
# Example news text
new_text = [test_x]

# Predict the encoded label
predicted_label_encoded = model.predict(new_text)

# Decode the label back to the original category name
predicted_category = le.inverse_transform(predicted_label_encoded)

print("Predicted Category:", predicted_category[0])

#%%
print(y[199701])
#%%
model = joblib.load('best_model.pkl')

# Recreate and fit the LabelEncoder on the original labels if not saved
# Assuming `y` is the original labels in your training data
le = LabelEncoder()
y = df['category_grouped']   # Assuming this is the original data
le.fit(y)
#%%
y
#%%
test_x = """With less than a week to go for campaigning to end in Maharashtra, we are well and truly into the last stretch of an Assembly election that, to a large extent, will determine if the Opposition has what it takes to sustain the sense of hope that animated it after the Lok Sabha polls or if the BJP-led NDA will be able to establish that the parliamentary elections were an aberration.

For the BJP, both its top two leaders will be in the state. Prime Minister Narendra Modi is scheduled to address public meetings in Chimur (1 pm) in eastern Vidarbha and Solapur (4.15 pm) and Pune (6.30 pm) in western Maharashtra.

Like in some other parts of the state, the Mahayuti government is also facing a degree of unrest among farmers in Vidarbha, which is the state’s cotton belt. How the PM tackles this in his speech will be among the things to watch out for as it will send a signal to farmers in the rest of the state dealing with agrarian distress.

In western Maharashtra that has 70 Assembly constituencies, the BJP and its Mahayuti allies will look to minimise any damage in what is essentially considered an NCP-Congress stronghold. Here, the BJP is looking to strategically deploy the PM to paper over any cracks in its organisation at the ground level.

The PM’s second-in-command and Union Home Minister Amit Shah will start his day in Jharkhand’s Dhanbad district by addressing rallies in Jharia Assembly constituency (11.30 am) and Baghmara Assembly seat (1.15 pm) before flying to Maharashtra to address public meetings in Ghatkopar East in suburban Mumbai (5.30 pm) and Borivali (7.30 pm)."""
#%%
print(has_emoji(test_x))
#%%
test_x = tokenize_text(test_x)
#%%
print(test_x)
#%%
categories = np.array(['national', 'entertainment', 'politics', 'world', 'technology',
       'business', 'sports', 'Health___Fitness', 'startup', 'automobile',
       'fashion', 'science', 'education', 'travel', 'miscellaneous',
       'hatke', 'Russia-Ukraine_Conflict', 'Coronavirus',
       'Asian_Games_2022', 'ODI_World_Cup_2023', 'Israel-Hamas_War',
       'myfeed', 'metaverse', 'policy', 'facts', 'Hockey_World_Cup_2023',
       'Feel_Good_Stories'], dtype='<U23')

#%%
y.to_csv('trans_data.csv')
#%%
y = pd.read_csv('trans_data.csv')
#%%
y
#%%
