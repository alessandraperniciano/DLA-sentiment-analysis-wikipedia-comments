# read csv file and save it
import pandas as pd
import numpy as np

import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.metrics as metrics
import xgboost as xgb


import matplotlib.pyplot as plt


#---------------------------------------------------------------#
def read_csv_file(file_name):
    data = pd.read_csv(file_name)
    return data

def clean_text(text):
    " Clean the text from special characters, links, punctuation and words with numbers in them "
    
    # Modify text in lower case
    text = str(text).lower()
    
    # Delete special characters
    text = re.sub('\[.*?\]', '', text)
    
    # Delete links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    # Delete punctuation
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    #text = re.sub('\n', '', text)
    
    # Delete words with numbers in them
    text = re.sub('\w*\d\w*', '', text)
    return text
    
def preprocessing_data(text):
    " Preprocess the data to be more readable for the model"
    
    # Download stopwords if not downloaded
    if stopwords is None:
        nltk.download('stopwords')
     
    # Clean text   
    text = clean_text(text)
    
    # Remove stop words
    stop_words = stopwords.words('english')
    more_stopwords = ['u', 'im', 'c']
    stop_words = stop_words + more_stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)

    # Stemming all words
    stemmer = nltk.SnowballStemmer("english")
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    
    return text

#-------------------------------------------------------#
    
if __name__ == '__main__':
    # Read csv files and save them in dataframes
    
    # labels = id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
    train = read_csv_file('data/train.csv')
   
    # labels = id, comment_text
    test = read_csv_file('data/test.csv')
    
    # labels = id, toxic, severe_toxic, obscene, threat, insult, identity_hate
    test_labels = read_csv_file('data/test_labels.csv')
    
    print("------------ Read csv files\n")
    
    # Clean all file
    train['comment_text_clean'] = train['comment_text'].apply(preprocessing_data)
    test['comment_text_clean'] = test['comment_text'].apply(preprocessing_data)
    
    print("------------ Cleaned files\n")


    # Vectorization
    x_train = train['comment_text_clean']
    y_train = train['toxic']
    
    x_test = test['comment_text_clean']
    y_test = test_labels['toxic']
    
    pipe = Pipeline([
        ('bow', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='auc',
            ))
    ])
    
    print("------------ Fit the model\n")
    pipe.fit(x_train, y_train)
    
    y_pred_class = pipe.predict(x_test)
    y_pred_train = pipe.predict(x_train)
    
    print("------------ Results\n")
    print('Train: {}'.format(metrics.accuracy_score(y_train, y_pred_train)))
    print('Test: {}'.format(metrics.accuracy_score(y_test, y_pred_class)))

    matrix = metrics.confusion_matrix(y_test, y_pred_class, labels = pipe.classes_)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=pipe.classes_)
    color = 'white'
    disp.plot()
    plt.xlabel('Predicted Label', color=color)
    plt.ylabel('True Label', color=color)
    plt.gcf().axes[0].tick_params(colors=color)
    plt.gcf().axes[1].tick_params(colors=color)
    plt.show()
    
    
    '''
    # Initialize the count vector
    vect = CountVectorizer()
    vect.fit(x_train)
    
    x_train_dtm = vect.transform(x_train)
    x_test_dtm = vect.transform(x_test)
    
    # Initialize the tfidf transformer
    tfidf = TfidfTransformer()
    tfidf.fit(x_train_dtm)
    x_train_tfidf = tfidf.transform(x_train_dtm)
    
    #Tokenization
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(x_train)
    vocab_length = len(word_tokenizer.word_index) + 1
    
    # Convert text to sequence
    
    x_train_seq = word_tokenizer.texts_to_sequences(x_train)
    longest_train = max(x_train, key=lambda sentence: len(sentence))
    len_long_train = len(word_tokenizer(longest_train))
    
    train_padded_sentences = pad_sequences(x_train_seq, maxlen=len_long_train, padding='post')
    '''