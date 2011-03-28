
import csv
import nltk
from nltk.stem.porter import PorterStemmer
import operator
import random
import string

#def extractWordCounts( comment ):
    

# Input:
# words = dictionary { stemmedWords -> word count }
# comment - lower case string with punctuation removed
# hr - helpfulness ratio
def comment_features(words,comment,hr):
    
    features={}
    
    # Break comment into list of words
    # TODO: This is already done previously in PrepareFeatureSets!
    tokenizedComment = nltk.word_tokenize(comment.lower()) 
    
    # Create list of stemmed words in the comment
    # TODO: Is this redundant as well? - previous stemmed only parts of speech
    stemmer = PorterStemmer()
    stemmedTokenizedComment = [stemmer.stem(word) for word in tokenizedComment]
    
    # TODO: This is done previously in last function
    # Create list of [word, pos] tuples
    posTokenizedComment = nltk.pos_tag(tokenizedComment);
            
    prevWord='$'
    for (word,type) in posTokenizedComment:
        
        # Catches phrases like "nice review"
        if(word.count('review')>0):
            features[prevWord + " review"]=1;
        # Converts phrases like "review [was] nice" into "nice review"
        elif(prevWord.count('review')>0):
            features[word + " review"]=1;
        
        # Catches phrases like "thank you"    
        # TODO: If we flip these statements, they can be factored into a function foo(word) with the section above
        if(prevWord.count('thank')>0):
            features["thank " + word]=1;
        # Catches phrases like "you [was] thank" and converts to "thank you"
        elif(word.count('thank')>0):
            features["thank " + prevWord]=1;
        
        prevWord=word
    
    for (word,count) in words:  
        if(stemmedTokenizedComment.count(word)>0):
            features[word]=1
            #features[word.lower()]=stemmedTokenizedComment.count(word.lower())
        else:
            features[word]=0
    
        
    return features

# Data is excel spreadsheet:
# List of dictionaries where each row is keyed by column label
def PrepareFeatureSets(data):
    stemmer = PorterStemmer()
    
    stopwords = nltk.corpus.stopwords.words('english')
    
    words = {}
    comments = []
    i = 0
    for n in data:
        # Replace punctuation with white space
        comment = n["Comment"].lower();
        for punct in string.punctuation:
            comment=comment.replace(punct," ")
        
        # Tokenize into list of words    
        tokenizedComment = nltk.word_tokenize(comment) 
        print "Parsing comment #" + str(i)
        
        # Creates a list of (word, part of speech) tuples
        posTokenizedComment = nltk.pos_tag(tokenizedComment);
        
        for word,part in posTokenizedComment:
            # If adjective or noun and not a stop word
            if((part=='JJ' or part=='NN') and word not in stopwords):
                # Stem the word
                stemmedWord = stemmer.stem(word)
                # Increment stemmed word count
                if(stemmedWord in words):
                    words[stemmedWord] += 1
                else:
                    print "Adding " + str(stemmedWord)
                    words[stemmedWord] = 1
        comments.append([comment,n["Thumbs Up!"],n["Helpfullness Ratio"]])
        i=i+1
        # END FOR LOOP
    
    # Sort by value (wordCount) rather than key (stemmedWords) - converts dictionary to list of tuples [ [stemmedWord, wordCount], ... ]
    sortedWords = sorted(words.iteritems(), key = operator.itemgetter(1))
    
    # TODO: extract into function
    # Extract only words between threshold count ranges
    threshMin=10
    threshMax=1100
    filteredWords = [ (w,n) for (w,n) in sortedWords if n>threshMin and n<threshMax]
    
    # Print filtered words
    i = 0
    for word in filteredWords:
        print str(i) + ": " + str(word)
        i += 1
    
    # Return 
    return [(comment_features(filteredWords,comment,hr), type) for (comment,type,hr) in comments]

if __name__ ==  "__main__":     
    data = csv.DictReader(open("../data/training-data.csv"))
    data = [n for n in data]
    data = data[0:100]
    
    featuresets = PrepareFeatureSets(data)
    
    random.shuffle(featuresets)
    
    train_set, test_set = featuresets[len(featuresets)/2:], featuresets[:len(featuresets)/2]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)
    
    classifier.show_most_informative_features(1000);
    
    #for fs in test_set:
    #    print str(fs) + "\r\nCLASS:" + str(classifier.prob_classify(fs));
        
    
    
    #classifier = nltk.DecisionTreeClassifier.train(train_set)
    #print nltk.classify.accuracy(classifier, test_set)
    #print classifier.most_informative_features(100);
    
    #classifier = nltk.classify.maxent.MaxentClassifier.train(train_set)
    #print nltk.classify.accuracy(classifier, test_set)
    
    
