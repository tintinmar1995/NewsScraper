import json
import nltk
import re
import sys
import os
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

from nltk.stem.snowball import FrenchStemmer

nltk.download('punkt')
nltk.download('stopwords')


synonyms = {'coronaviru':'covid',
            'covid19':'covid'}

def run(stopWords, stemmer, lang='fr'):
    global synonyms

    # read file
    with open('scraped_articles.json', 'r') as myfile:
        data=myfile.read()
    # parse file
    obj = json.loads(data)

    titles = []
    original_titles = []
    for newspaper in obj['newspapers']:
        for article in obj['newspapers'][newspaper]['articles']:
            tokenized_title = []
            original_titles.append(article['title'])
            pending_title = re.sub('[^A-Za-zàâçéèêëîïôûùüÿñæœ0-9 ]+', ' ', article['title'])
            pending_title.replace("  ", " ")
            for w in nltk.tokenize.word_tokenize(pending_title):
                if w not in stopWords and len(w)>3:
                    w = stemmer.stem(w)
                    w = synonyms[w] if w in synonyms.keys() else w
                    tokenized_title.append(w)
            titles.append(tokenized_title)
            
    vectorizer = CountVectorizer(ngram_range=(1,1))
    matrix = vectorizer.fit_transform([" ".join(title) for title in titles])
    return original_titles, KMeans(n_clusters=10).fit_predict(matrix)
    
def main():

    args = list(sys.argv)

    lang = 'en'
    if "--lang" in args:
        idx = args.index("--lang")
        lang = str(args[idx + 1])
        args = [args[i] for i in range(len(args)) if i not in (idx, idx + 1)]

    if lang=='en':
        stopWords = set(nltk.corpus.stopwords.words('english'))
        stemmer = nltk.stem.snowball.PorterStemmer()
        #os.system("python newsscraper.py NewsPapers.json")
    elif lang=='fr':
        stopWords = set(nltk.corpus.stopwords.words('french'))
        stemmer = nltk.stem.snowball.FrenchStemmer()
        #os.system("python newsscraper.py FrenchPapers.json")
        
    original_titles, preds = run(stopWords, stemmer, lang)

    for grp in set(preds):
        print("")
        print("---------------")
        print(np.array(original_titles)[preds==grp])
        print("---------------")
        print("")
    
if __name__ == "__main__":
    main()
