from collections import Counter

import nltk
import matplotlib
from nltk.tokenize import word_tokenize
from nltk.lm.vocabulary import Vocabulary
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def a(df):
    # A)

    words = [word for sentence in df['word_token'] for word in sentence]
    # Reading politeness

    # Tokenizing all words in every sentence
    politeness['word_token'] = politeness['Sentence'].apply(word_tokenize)
    # print(politeness)

    # Vocabulary
    # First I collapse all tokenized words into one list
    df = politeness.copy()
    vocabulary = Vocabulary(words, 1)

    # Vocabulary count returns a Collections.Counter object that has the counts of every word
    # print('Word counts: ' + str(vocabulary.counts))
    # Total count of diferent words
    print('There are this many words in the vocabulary: ' + str(len(vocabulary.counts.keys())))

    data = vocabulary.counts.most_common(10)

    # COMMENTED OUT
    bar = px.bar(
        x=[x for (x, y) in data],
        y=[y for (x, y) in data]
    )

    bar.show()

    # Punctuation and stopwords are the most common


def b(df):
    # B)

    words = [word for sentence in df['word_token'] for word in sentence]
    # Removing punctuation and stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = \
        [word for word in words
         if word not in stop_words
         and word.isalnum()
         ]
    filtered_vocabulary = Vocabulary(filtered_words, 1)

    # print('Word counts of filtered vocabulary: ' + str(filtered_vocabulary.counts))

    # Total count of diferent words
    print('There are this many words in the filtered vocabulary: ' + str(len(filtered_vocabulary.counts.keys())))

    data1 = filtered_vocabulary.counts.most_common(10)

    # COMMENTED OUT
    # Ne znam zosto ne mi go prikazhuva bagnato e izgleda
    bar = px.bar(
        x=[x for (x, y) in data1],
        y=[y for (x, y) in data1]
    )

    bar.show()


def c(df):
    # C
    words = [word for sentence in df['word_token'] for word in sentence]
    # Removing punctuation and stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = \
        [word for word in words
         if word not in stop_words
         and word.isalnum()
         ]

    lemmitizer = WordNetLemmatizer()
    lemmatized_words = [lemmitizer.lemmatize(word) for word in filtered_words]
    lemmatized_vocabulary = Vocabulary(lemmatized_words, 1)
    datal = lemmatized_vocabulary.counts.most_common(10)

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    stemmed_vocabulary = Vocabulary(stemmed_words)
    datas = stemmed_vocabulary.counts.most_common(10)

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Bar(
            x=[x for (x, y) in datas],
            y=[y for (x, y) in datas]
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=[x for (x, y) in datal],
            y=[y for (x, y) in datal]
        ),
        row=1,
        col=2
    )

    fig.show()

    # Total count of diferent words
    print('There are this many words in the stemmed vocabulary: ' + str(len(stemmed_vocabulary.counts.keys())))

    # Total count of diferent words
    print('There are this many words in the lemmatized vocabulary: ' + str(len(lemmatized_vocabulary.counts.keys())))


if __name__ == '__main__':
    politeness = pd.read_csv("train_en.txt", sep='\t')
    politeness['word_token'] = politeness['Sentence'].apply(word_tokenize)


    while True:
        print('Choose A, B or C for Problem 1')
        print('Enter exit to exit')
        option = input()
        if option.lower() == 'a':
            a(politeness)
        elif option.lower() == 'b':
            b(politeness)
        elif option.lower() == 'c':
            c(politeness)
        elif option.lower() == 'exit':
            break
        else:
            print("Please choose a valid option")
