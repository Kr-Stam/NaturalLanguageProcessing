import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
from gensim import downloader

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


def a():
    word2Vec = getModel()


    while True:
        print("Enter positive and negative words to find the most similar word")
        pos_words = input("\tEnter positive words\n").split(' ')
        neg_words = input("\tEnter negative words\n").split(' ')

        if neg_words == ['']:
            neg_words = []
        if pos_words == ['']:
            pos_words = []
        try:
            print('\t' + str(word2Vec.wv.most_similar(positive=pos_words, negative=neg_words)))
        except:
            print("\tYou entered words that do not exist in the training set")


def getModel() -> Word2Vec:
    try:
        word2Vec = Word2Vec.load("word2vec.model")
    except:
        word2Vec = generateModel()
        word2Vec.save("word2vec.model")
    return word2Vec


def b():
    embeddings = downloader.load('glove-twitter-50')

    # Paris – France + Italy = Rome
    print("Paris – France + Italy = Rome")
    print(embeddings.most_similar(positive=['france', 'italy'], negative='paris'))
    # Madrid – Spain + France = Paris
    print("Madrid – Spain + France = Paris")
    print(embeddings.most_similar(positive=['madrid', 'france'], negative='italy'))
    # King – Man + Woman = Queen
    print("King – Man + Woman = Queen")
    print(embeddings.most_similar(positive=['king', 'woman'], negative='man'))
    # Bigger – Big + Cold = Colder
    print("Bigger – Big + Cold = Colder")
    print(embeddings.most_similar(positive=['bigger', 'cold'], negative='big'))
    # Windows – Microsoft + Google = Android
    print("Windows – Microsoft + Google = Android")
    print(embeddings.most_similar(positive=['windows', 'google'], negative='microsoft'))


if __name__ == '__main__':

    option = input()

    while True:
        print("Choose A or B for Problem2")
        print("Enter exit to exit")
        if option.lower() == 'a':
            a()
        elif option.lower() == 'b':
            b()
        elif option.lower() == 'exit':
            break;
        else:
            print("Please choose a valid option")


