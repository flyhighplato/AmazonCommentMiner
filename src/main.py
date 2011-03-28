
import csv
import nltk
from nltk.stem.porter import PorterStemmer
import operator
import random
import string

#def extractWordCounts( comment ):
    
    
def comment_features(words,comment,hr):
    features={}
    tokenizedComment = nltk.word_tokenize(comment.lower()) 
    stemmer = PorterStemmer()
    
    stemmedTokenizedComment = [stemmer.stem(word) for word in tokenizedComment]
    stopwords = nltk.corpus.stopwords.words('english')
    #features["length"]=len(comment)
    
    posTokenizedComment = nltk.pos_tag(tokenizedComment);
        
    oldword='$'
    for (word,type) in posTokenizedComment:
        if(word.count('review')>0):
            features[oldword + " review"]=1;
        elif(oldword.count('review')>0):
            features[word + " review"]=1;
        if(oldword.count('thank')>0):
            features["thank " + word]=1;
        elif(word.count('thank')>0):
            features["thank " + oldword]=1;
        oldword=word
    
    for (word,count) in words:  
        if(word.lower() not in stopwords):
            if(stemmedTokenizedComment.count(word.lower())>0):
                features[word.lower()]=1
                #features[word.lower()]=stemmedTokenizedComment.count(word.lower())
            else:
                features[word.lower()]=0
                
    if(hr>=0 and hr<0.34):
        features["HR"]='low'
    elif(hr>=0.34 and hr<0.5):
        features["HR"]='mlow'
    elif(hr>=0.5 and hr<0.67):
        features["HR"]='mhigh'
    else:
        features["HR"]='high'
        
    return features

if __name__ ==  "__main__":     
    data = csv.DictReader(open("../data/training-data.csv"))
    stemmer = PorterStemmer()
    
    stopwords = nltk.corpus.stopwords.words('english')
    
    words = {}
    comments = []
    i = 0
    for n in data:
        comment = n["Comment"].lower();
        for punct in string.punctuation:
            comment=comment.replace(punct," ")
            
        tokenizedComment = nltk.word_tokenize(comment) 
        print "Parsing comment #" + str(i)
        
        posTokenizedComment = nltk.pos_tag(tokenizedComment);
        
        for word,part in posTokenizedComment:
            if((part=='JJ' or part=='NN') and word not in stopwords):
                #print "Adding " + word
                stemmedWord = stemmer.stem(word)
                if(stemmedWord in words):
                    words[stemmedWord] += 1
                else:
                    print "Adding " + str(stemmedWord)
                    words[stemmedWord] = 1
        comments.append([comment,n["Thumbs Up!"],n["Helpfullness Ratio"]])
        i=i+1

    sortedWords = sorted(words.iteritems(), key = operator.itemgetter(1))
    threshMin=10
    threshMax=1100
    filteredWords = [ (w,n) for (w,n) in sortedWords if n>threshMin and n<threshMax]
    
    
    i = 0
    for word in filteredWords:
        print str(i) + ": " + str(word)
        i += 1
    
    featuresets = [(comment_features(filteredWords,comment,hr), type) for (comment,type,hr) in comments]
    random.shuffle(featuresets)
    
    train_set, test_set = featuresets[len(featuresets)/2:], featuresets[:len(featuresets)/2]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)
    classifier.show_most_informative_features(100);
    
    #classifier = nltk.DecisionTreeClassifier.train(train_set)
    #print nltk.classify.accuracy(classifier, test_set)
    #print classifier.most_informative_features(100);
    
    #classifier = nltk.classify.maxent.MaxentClassifier.train(train_set)
    #print nltk.classify.accuracy(classifier, test_set)
    
    
