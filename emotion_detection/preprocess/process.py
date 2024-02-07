import pandas as pd
from hazm import Stemmer, word_tokenize, Normalizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from cleantext import clean
import re
from parsivar import Normalizer




def cleaning(text):
    text = text.strip()

    # regular cleaning
    text = clean(text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="0",
        replace_with_currency_symbol="",
    )


    # normalizing
    normalizer = Normalizer()
    text = normalizer.normalize(text)

    # removing wierd patterns
    wierd_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        # u"\u200c"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)

    text = wierd_pattern.sub(r'', text)

    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)

    return text


def prepared_data(path):

    df = pd.read_excel(path, names=['ID', 'text1', 'text2', 'emotion'])
    # some preprocess on
    filled_df = df.groupby('ID').apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    # # Select nan variable emotion column
    # filled_df.loc[filled_df['emotion'].isnull()]
    # fill nan variable in text1 and text2 with ''
    filled_df[['text1', 'text2']].fillna('', inplace=True)
    # drop nan variable in axis=0
    filled_df.dropna(axis=0, subset=['ID', 'emotion'], inplace=True)
    filled_df.dropna(inplace=True)
    # New df
    filled_df.groupby(['ID', 'emotion'])['text1', 'text2'].agg(' '.join).reset_index()
    # merge df
    # combine text1 and text2 columns
    filled_df['text1'] = filled_df['text1'].astype(str)
    filled_df['text2'] = filled_df['text2'].astype(str)
    filled_df['combined_text'] = filled_df['text1'] + ' ' + filled_df['text2']
    new_df = filled_df.copy()
    new_df.drop(['text1', 'text2'], axis=1, inplace=True)
    # combine with groupby ID and emotion
    merged_df = new_df.groupby(['ID', 'emotion']).agg(lambda x: ' '.join(x)).reset_index()
    # delete ID column
    merged_df.drop('ID', axis=1, inplace=True)
    return merged_df


def preprocess_text(df):
    def process_text(text):
      text = re.sub(r'[^ا-ی!\s]', '', text)
      text = re.sub('[0-9]','',text)
    
    #   stemmer = Stemmer()
    #   text_tokens = word_tokenize(text)
    #   texts_clean = [stemmer.stem(word) for word in text_tokens]
    #   text = " ".join(texts_clean)
      return text
    df.combined_text = df.combined_text.apply(process_text)
    df.to_csv('data/preprocess_data.csv', index=False)
    return df

def preprocess_ArmanEmo(df):
    def process_text(text):
      text = re.sub(r'[^ا-ی!\s]', '', text)
      text = re.sub('[0-9]','',text)
      text = re.sub(r'\s*[A-Za-z]+\b', '' , text)

      chars_to_remove = "؟«»٪۰۱۲۳۴۵۶۷۸۹"
      regex_remove = f"[{chars_to_remove}]"
      text = re.sub(regex_remove, '', text)

      my_normalizer = Normalizer()
      tmp_text = my_normalizer.normalize(text)
      return tmp_text
    df.text = df.text.apply(process_text)
    cleaned_df = df.dropna()
    cleaned_df.to_csv('data/train_process.csv', index=False)
    return cleaned_df

def tfidf_vectorizer(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf



def process_length_words(data):
    data['text_len_by_words'] = data.combined_text.apply(lambda x: len(str(x).split()))
    data['tokens'] = data.combined_text.apply(lambda x: str(x).split())

    # remove length more than 450
    data = data[data['text_len_by_words']<=450]
    data.drop('text_len_by_words', axis=1, inplace=True)
    return data