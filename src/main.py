
import java
import javax.swing as swing
import java.awt as awt
import csv
import nltk
from nltk.stem.porter import PorterStemmer
import operator
import random
#def extractWordCounts( comment ):
    


def appMain():
    win = swing.JFrame("Jython", size=(200, 200),windowClosing=exit)
    win.contentPane.layout = awt.FlowLayout(  )
    
    field = swing.JTextField(preferredSize=(200,20))
    field.setText("Hello world!")
    field.setEnabled(False);
    
    win.contentPane.add(field)
    
    win.pack()
    win.show()
    
def comment_features(words,comment):
    features={}
    tokenizedComment = nltk.word_tokenize(comment.lower()) 
    stemmer = PorterStemmer()
    
    stemmedTokenizedComment = [stemmer.stem(word.lower()) for word in tokenizedComment]
    stopwords = nltk.corpus.stopwords.words('english')
    #features["length"]=len(comment)
    for (word,count) in words:
        
        if(word.lower() not in stopwords):
            if(stemmedTokenizedComment.count(word.lower())>0):
                features[word.lower()]=1
            else:
                features[word.lower()]=0
            
    return features

if __name__ ==  "__main__":     
    #appMain()
    print("Hello world")
    data = csv.DictReader(open("../data/training-data.csv"))
    #featuresets = [(comment_features(n["Comment"]), n["Thumbs Up!"]) for n in data]
    #for n in data:
    stemmer = PorterStemmer()
    
    stopwords = nltk.corpus.stopwords.words('english')
    
    words = {}
    comments = []
    for n in data:
        tokenizedComment = nltk.word_tokenize(n["Comment"].lower()) 
        for word in tokenizedComment:
            if(word not in stopwords):
                stemmedWord = stemmer.stem(word)
                if(stemmedWord in words):
                    words[stemmedWord] += 1
                else:
                    words[stemmedWord] = 1
        comments.append([n["Comment"],n["Thumbs Up!"]])

    sortedWords = sorted(words.iteritems(), key = operator.itemgetter(1))
    threshMin=20
    threshMax=100
    filteredWords = [ (w,n) for (w,n) in sortedWords if n>threshMin and n<threshMax]
    
    
    i = 0
    for word in filteredWords:
        print str(i) + ": " + str(word)
        i += 1
    
    featuresets = [(comment_features(filteredWords,comment), type) for (comment,type) in comments]
    random.shuffle(featuresets)
    
    #print featuresets
    
    train_set, test_set = featuresets[len(featuresets)/2:], featuresets[:len(featuresets)/2]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)
    
    
