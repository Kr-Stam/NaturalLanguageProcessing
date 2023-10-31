import sklearn
from gensim.models import Word2Vec
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import plotly.express as px
from scipy.stats import kendalltau, pearsonr
import numpy as np


def load_data() -> pd.DataFrame:
    return pd.read_csv('train_en.txt', sep='\t')


def tokenize(df: pd.DataFrame):
    df['word_tokens'] = df['Sentence'].apply(word_tokenize)


def generateModel() -> Word2Vec:
    global word2Vec
    politeness = load_data()
    tokenize(politeness)
    sentences = politeness['word_tokens'].values
    # sg=1 means SkipGram
    # sg=0 means ContinousBagOfWords
    word2Vec = Word2Vec(sentences, vector_size=50, min_count=15, window=3, sg=1, workers=4)
    return word2Vec


def getModel() -> Word2Vec:
    try:
        word2Vec = Word2Vec.load("word2vec.model")
    except:
        word2Vec = generateModel()
        word2Vec.save("word2vec.model")
    return word2Vec


# Ne gledav prichina da kompliciram megju da gi delam na posebni delovi a i b
def solution():
    print("Start of A")
    global word2VecProblem2, pairs
    word2VecProblem2 = getModel()
    pairs = pd.read_csv("wordsim353/combined.csv", sep=',')
    pairs = pairs[pairs['Word 1'].isin(word2VecProblem2.wv.key_to_index.keys())]
    pairs = pairs[pairs['Word 2'].isin(word2VecProblem2.wv.key_to_index.keys())]

    e_predictions = []
    c_predictions = []
    for word1, word2 in zip(pairs['Word 1'], pairs['Word 2']):
        vector1 = word2VecProblem2.wv[word1]
        vector2 = word2VecProblem2.wv[word2]
        # print(cosine_similarity([vector1], [vector2]))
        c_predictions.extend(cosine_similarity([vector1], [vector2])[0])
        # print(1 - euclidean_distances([vector1], [vector2]))
        e_predictions.extend(1 - euclidean_distances([vector1], [vector2])[0])

    pairs['e_predict'] = e_predictions
    pairs['c_predict'] = c_predictions

    print("Saved to csv")
    pairs.to_csv("test.csv")

    print("Start of B")
    print("The coefficients of similarity for the cosine predictions are")
    r, p = pearsonr(pairs['Human (mean)'], pairs['c_predict'])
    print("r=" + str(r))
    print("p=" + str(p))

    print("The coefficients of similarity for the euclidian predictions are")
    r, p = pearsonr(pairs['Human (mean)'], pairs['e_predict'])
    print("r=" + str(r))
    print("p=" + str(p))


if __name__ == '__main__':
    print("Tuka posledovatelno se reshava a i b")

    solution()
