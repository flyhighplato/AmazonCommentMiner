
import csv
import nltk
from nltk.stem.porter import PorterStemmer
import operator
import math
import string
import random

#def extractWordCounts( comment ):
    
stopwords = nltk.corpus.stopwords.words('english')
# Input:
# words = dictionary { stemmedWords -> word count }
# comment - lower case string with punctuation removed
# hr - helpfulness ratio
def comment_features(words,comment,hr):
    
    features={}
    if(len(comment)>100):
        features["LENGTH"]="HIGH"
    else:
        features["LENGTH"]="LOW"
        
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
          
    rawWords = [w for (w,c) in words]
      
    prevWord='$'
    for (word,part) in posTokenizedComment:
        if((part.startswith('JJ') or part.startswith('NN')) and word not in stopwords):
            stemmedWord = stemmer.stem(word)
            if(stemmedWord in rawWords and prevWord in rawWords):
                phrase1 = prevWord + " " + stemmedWord
                phrase2 = stemmedWord + " " + prevWord
                
                defaultPhrase = phrase1
                if(phrase2 in features.keys()):
                    defaultPhrase = phrase2
                
                features[defaultPhrase]=1
                
            # Catches phrases like "nice review"
            #if(word.count('review')>0):
            #    features[prevWord + " review"]=1;
            # Converts phrases like "review [was] nice" into "nice review"
            #elif(prevWord.count('review')>0):
            #    features[word + " review"]=1;
            
            # Catches phrases like "thank you"    
            # TODO: If we flip these statements, they can be factored into a function foo(word) with the section above
            #if(prevWord.count('thank')>0):
            #    features["thank " + word]=1;
            # Catches phrases like "you [was] thank" and converts to "thank you"
            #elif(word.count('thank')>0):
            #    features["thank " + prevWord]=1;
            
            prevWord=stemmedWord
    
    for (word,count) in words:  
        if(stemmedTokenizedComment.count(word)>0):
            features[word]=1
            #features[word.lower()]=stemmedTokenizedComment.count(word.lower())
        else:
            features[word]=0
    hr = float(hr)       
    if(hr>0 and hr<0.2):
        features["HR"]="LOW"
    elif(hr>=0.2 and hr<0.5):
        features["HR"]="MLOW"
    elif(hr>=0.5 and hr<0.8):
        features["HR"]="MHIGH"
    elif(hr<1 and hr>=0.8):
        features["HR"]="HIGH"
    
        
    return features

# Data is excel spreadsheet:
# List of dictionaries where each row is keyed by column label
def PrepareFeatureSets(data):
    stemmer = PorterStemmer()
    
    
    
    reviewIds = set()
    wordsPerReview = {}
    words = {}
    comments = []
    i = 0
    for n in data:
        # Replace punctuation with white space
        comment = n["Comment"].lower();

        reviewIds.update([n["Review_ID"]])

        for punct in string.punctuation:
            comment=comment.replace(punct," ")
        
        # Tokenize into list of words    
        tokenizedComment = nltk.word_tokenize(comment) 
        print "Parsing comment #" + str(i)
        
        # Creates a list of (word, part of speech) tuples
        posTokenizedComment = nltk.pos_tag(tokenizedComment);
        
        for word,part in posTokenizedComment:
            # If adjective or noun and not a stop word
            if((part.startswith('JJ') or part.startswith('NN')) and word not in stopwords):
                # Stem the word
                stemmedWord = stemmer.stem(word)
                # Increment stemmed word count
                if(stemmedWord in words):
                    words[stemmedWord] += 1
                    wordsPerReview[stemmedWord].update([n["Review_ID"]])
                    #print wordsPerReview[stemmedWord]
                else:
                    print "Adding " + str(stemmedWord)
                    words[stemmedWord] = 1
                    wordsPerReview[stemmedWord]=set([n["Review_ID"]])
                    #print wordsPerReview[stemmedWord]
        comments.append([comment,n["Thumbs Up!"],n["Helpfullness Ratio"]])
        i=i+1
        # END FOR LOOP
    
    # Sort by value (wordCount) rather than key (stemmedWords) - converts dictionary to list of tuples [ [stemmedWord, wordCount], ... ]
    sortedWords = sorted(words.iteritems(), key = operator.itemgetter(1))
    
        
    # TODO: extract into function
    # Extract only words between threshold count ranges
    threshMin=10
    threshMax=1100
    filteredWords = [ (w,n) for (w,n) in sortedWords if n>threshMin and n<threshMax and w in wordsPerReview and ( float(len(wordsPerReview[w]))/ float(len(reviewIds)) )>0.3 ]
    
    # Print filtered words
    i = 0
    
    
    print ""
    for word in filteredWords:
        if word[0] in wordsPerReview.keys():
            print str(i) + ": " + str(word) + ":" + str(float(len(wordsPerReview[word[0]]))/ float(len(reviewIds)))
        else:
            print str(i) + ": " + str(word) + ": NOT THERE!"
        i += 1
    
    # Return 
    return [(comment_features(filteredWords,comment,hr), type) for (comment,type,hr) in comments]

# Only parts of speech for now
def getSvmFeatureSet( filteredWords, comment, hr ):
    features={}
    
    # Break comment into list of words
    # TODO: This is already done previously in PrepareFeatureSets!
    tokenizedComment = nltk.word_tokenize(comment.lower()) 
    
    # Create list of stemmed words in the comment
    # TODO: Is this redundant as well? - previous stemmed only parts of speech
    stemmer = PorterStemmer()
    stemmedTokenizedComment = [stemmer.stem(word) for word in tokenizedComment]
    
    for (word,idf) in filteredWords:  
        if(stemmedTokenizedComment.count(word)>0):
            features[word]=float(stemmedTokenizedComment.count(word))
            #features[word.lower()]=stemmedTokenizedComment.count(word.lower())
        else:
            features[word]=0.0
        features[word]=features[word]/float(len(stemmedTokenizedComment))*idf
   
    hr = float(hr)       
    if(hr>0 and hr<0.2):
        features["HR"]=-1
    elif(hr>=0.2 and hr<0.5):
        features["HR"]=-0.5
    elif(hr>=0.5 and hr<0.8):
        features["HR"]=0.5
    else: # if(hr<1 and hr>=0.8):
        features["HR"]=1

    return features;

# Data is excel spreadsheet:
# List of dictionaries where each row is keyed by column label
def PrepareSvmFeatureSets(data):
    stemmer = PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    
    words = {}
    comments = []
    
    i = 0
    IDF={}
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
                    if(word not in IDF.keys()):
                        IDF[word]=1
                    else:
                        IDF[word]+=1
        
        comments.append([comment,n["Thumbs Up!"],n["Helpfullness Ratio"]])
        i=i+1
        # END FOR LOOP
    
    for term in IDF:
        words[term]=math.log(float(len(data))/float(IDF[term]))
        
    # Sort by value (wordCount) rather than key (stemmedWords) - converts dictionary to list of tuples [ [stemmedWord, wordCount], ... ]
    sortedWords = sorted(words.iteritems(), key = operator.itemgetter(1))
    
    # TODO: extract into function
    # Extract only words between threshold count ranges
    #threshMin=2
    #threshMax=1000
    #filteredWords = [ (w,n) for (w,n) in sortedWords if n>threshMin and n<threshMax]
    filteredWords = sortedWords[500:]
    
    # Print filtered words
    i = 0
    for word in filteredWords:
        print str(i) + ": " + str(word)
        i += 1
    
    # Return 
    return [(getSvmFeatureSet(filteredWords,comment,hr), type) for (comment,type,hr) in comments]
    
def getSign( value ):
    if ( value >= 0 ):
        return "+"
    return ""

# Feature set is list of tuples of (dictionary, type)
def writeSvmFeatureSet( outFileName, featureSets ):
    
    file = open( outFileName, "w" )
    
    keyList = featureSets[ 0 ][ 0 ].keys()
    
    for ( featureSet, type ) in featureSets:
        # Output type
        svmType = -1 + 2*int(type)
        file.write( getSign(svmType) + str(svmType) )
        i=1
        for key in keyList:
            value = featureSet[ key ] 
            file.write( " " + str(i) + ":" + str(value) )
            i += 1
        file.write( " \n" )
        
if __name__ ==  "__main__":
    
    # Load training data
    data = csv.DictReader(open("../data/training-data.csv"))
    data = [n for n in data]
    random.shuffle(data)
    #data = data[0:1000]
    
# BEGIN BAYESIAN CLASSIFIER
    featuresets = PrepareFeatureSets(data)
    
    random.shuffle(featuresets)
    
    train_set, test_set = featuresets[len(featuresets)/2:], featuresets[:len(featuresets)/2]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)

    classifier.show_most_informative_features(1000);
# END BAYESIAN CLASSIFIER
    
# BEGIN SVM INTEGRATION
    #featureSets = PrepareSvmFeatureSets( data )
    
    #train_set, test_set = featureSets[len(featureSets)/2:], featureSets[:len(featureSets)/2]
    
    #writeSvmFeatureSet( "train.svm", train_set )
    #writeSvmFeatureSet( "test.svm", test_set )
    print "done\n"
# END SVM INTEGRATION
    
# Uncomment this out to classifiy an additional file
    #validationData = csv.DictReader(open("../data/validation-data.csv"))
    #validationFeatureSets = PrepareFeatureSets( validationData )
    #print nltk.classify.accuracy(classifier, validationFeatureSets)
    
    #for fs in test_set:
    #    print str(fs) + "\r\nCLASS:" + str(classifier.prob_classify(fs));
        
    
    
    #classifier = nltk.DecisionTreeClassifier.train(train_set)
    #print nltk.classify.accuracy(classifier, test_set)
    #print classifier.most_informative_features(100);
    
    #classifier = nltk.classify.maxent.MaxentClassifier.train(train_set)
    #print nltk.classify.accuracy(classifier, test_set)
    
    
