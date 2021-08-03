# Write your code here
from lxml import etree
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter

# nltk.download('averaged_perceptron_tagger')

xml_path = "news.xml"

tree = etree.parse(xml_path)
root = tree.getroot()

stop_words = set(stopwords.words('english'))
stop_words.update(string.punctuation)

lemmatizer = WordNetLemmatizer()

if __name__ == '__main__':
    all_stories = []

    for i, news in enumerate(root[0]):
        # print(news[0].text+':')  # Title
        wt = word_tokenize(news[1].text.lower())
        lemm = [lemmatizer.lemmatize(word) for word in wt]
        cleared_lemm = [word for word in lemm if word not in stop_words]
        nouns = [el for el in cleared_lemm if nltk.pos_tag([el])[0][1] == "NN"]
        all_stories.append(' '.join(nouns))


    vectorizer = TfidfVectorizer()  # vocabulary=set(nouns))
    tfidf_matrix = vectorizer.fit_transform(all_stories)
    terms = vectorizer.get_feature_names()
    for i, news in enumerate(root[0]):
        print(news[0].text + ':')  # Title

        d = [(terms[k], value) for k, value in enumerate(tfidf_matrix.toarray()[i])]
        j = 0
        for k in sorted(d, key=itemgetter(1, 0), reverse=True):
            if j < 5:
                print(k[0], end=' ')
                j += 1
                
        print()
        print()


