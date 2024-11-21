from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# buat nentuin jenis kata yang dimengerti wordnet lemmatizer
def get_wordnet_pos(tag):
    if tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('J'):
        return wn.ADJ
    else:
        return wn.NOUN

# data cleaning
def clean_text(text):
    # hapus tanda baca, tag html, dll
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    text = text.lower()

    # ambil karakter a-z & A-Z
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    return text

# data preprocessing
def preprocess_text(text):
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    pos_tags = pos_tag(words)

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    processed_text = ' '.join(words)

    return processed_text

def process_data(data_folder):
    processed_data = []

    # regex pattern
    pattern = re.compile(r'([a-zA-Z]+)_(\d+)')

    for root, dirs, files in os.walk(data_folder):
        for file in files:
            match = pattern.match(file)
            if match:
                # baca file
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    cleaned_content = clean_text(content)
                    preprocessed_content = preprocess_text(cleaned_content)

                    processed_data.append({'filename': file, 'processed_content': preprocessed_content, 'filepath': file_path})

    return pd.DataFrame(processed_data)


# path ke folder dataset
data_folder = 'data'

df = process_data(data_folder)

# buat kolom baru
df['clean_keyword'] = df['processed_content']
df['clean_keyword'] = df['clean_keyword'].apply(lambda x: "'" + ','.join(x.split()) + "'")

# cek keyword yang udah diclean
# print(df['clean_keyword'])

vocabulary = set()

for doc in df.clean_keyword:
    vocabulary.update(doc.split(','))

vocabulary = list(vocabulary)

# tf-idf
tfidf = TfidfVectorizer(vocabulary=vocabulary,dtype=np.float32)
tfidf.fit(df.clean_keyword)
tfidf_tran=tfidf.transform(df.clean_keyword)

# cek beberapa vocabulary
# vocabulary[0:5]

# vector untuk query
def gen_vector_T(tokens):
    # vektor kosong buat simpan vocabulary
    Q = np.zeros((len(vocabulary)))

    x= tfidf.transform(tokens)
    for token in tokens[0].split(','):
        # buat handling vocabulary
        try:
            ind = vocabulary.index(token)
            Q[ind]  = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return Q
    
def cosine_similarity_T(k, query):
    lemmatizer = WordNetLemmatizer()

    # Preprocessing
    preprocessed_query = re.sub("\W+", " ", query).strip()
    tokens = word_tokenize(str(preprocessed_query))
    
    # Lemmatize token-tokennya
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # diubah jadi string
    query_str = ' '.join(lemmatized_tokens)
    
    # dataframe buat query
    q_df = pd.DataFrame({'q_clean': [query_str]})

    # generate query vectornya
    query_vector = tfidf.transform(q_df['q_clean']).toarray()[0]

    # hitung cosine similarity di semua dokumen
    d_cosines = [cosine_similarity(query_vector.reshape(1, -1), d.toarray().reshape(1, -1)).flatten()[0] for d in tfidf_tran]

    # ambil index top-k documents
    result_indices = np.array(d_cosines).argsort()[-k:][::-1]
    d_cosines.sort()

    # buat dataframe untuk simpan hasilnya
    result_df = pd.DataFrame(columns=['index', 'filename', 'filepath', 'Score'])

    for i, index in enumerate(result_indices):
        result_df.loc[i, 'index'] = str(index)
        result_df.loc[i, 'filename'] = df['filename'][index]
        result_df.loc[i, 'filepath'] = df['filepath'][index]
        
        result_df.loc[i, 'Score'] = d_cosines[-k:][::-1][i]

    return result_df

# cosine_similarity_T(10,'computer science')