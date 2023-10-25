# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import nltk as nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # print(stopwords.words('english'))
    words = word_tokenize("This is a sentence. And this is another sentence")
    stemmer = PorterStemmer()
    print(stemmer.stem("eating"))
    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize("park", 'v'))
    # print(pos_tag("This is a sentence"))
    # nltk.download()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
