'''
MinerContext.py

@author: garyturovsky
@author: alanperezrathke
'''

import csv
import logging
import MinerMiscUtils
import nltk
from nltk.stem.porter import PorterStemmer
import random
import string

# Structure for easier passing around of "global" parameters
class Context:
    
    # Constructor
    def __init__( self, strPathToRawCsvComments, strPathToRawCsvReviews, filterWordCountMin, filterWordCountMax, filterWordReviewOverlap ):
        # Log parameters
        logging.getLogger("Context").info( "Creating new context:" )
        logging.getLogger("Context").info( "strPathToRawCsvComments: " + strPathToRawCsvComments )
        logging.getLogger("Context").info( "filterWordCountMin: " + str(filterWordCountMin) )
        logging.getLogger("Context").info( "filterWordCountMax: " + str(filterWordCountMax) )
        logging.getLogger("Context").info( "filterWordReviewOverlap: " + str(filterWordReviewOverlap) )
        
        # Load stop words
        self.mStopWords = nltk.corpus.stopwords.words('english')
        
        self.mRawCsvReviews = csv.DictReader(open(strPathToRawCsvReviews))
        self.mRawCsvReviews = [review for review in self.mRawCsvReviews]
                                                              
        # Load and shuffle comment data
        self.mRawCsvComments = csv.DictReader(open(strPathToRawCsvComments))
        self.mRawCsvComments = [comment for comment in self.mRawCsvComments]
        random.shuffle(self.mRawCsvComments)
        #self.mRawCsvComments=self.mRawCsvComments[0:100]
        
        # Parallel list of lower case comments with punctuation removed
        self.mLowerCasePunctRemovedComments = []
        
        # Parallel list for storing [ (word, part-of-speech ) ] tuple lists for each comment
        self.mPartOfSpeechTokenizedComments = []
        
        # Parallel list for storing [ stemmed(word) ] lists for each comment
        self.mStemmedTokenizedComments = []
        
        # Set for storing unique review identifiers
        self.mReviewIds = set()
        
        # Maps a stemmed word to a set of reviews it belongs to (for filtering kindle, ipod, etc)
        self.mStemmedWordToReviewsMap = {}
        
        # Create stemmer for stemming words
        stemmer = PorterStemmer()
        
        # Dictionary for storing word counts of adjectives and nouns
        self.mAdjAndNounWordCountMap = {}
        
        # Dictionary for storing custom data specific to a classifier
        self.mCustomData = {}
        
        self.mPartOfSpeechTokenizedCommentsAndReviewId = []
        
        self.mAuthorFreqPerReview={}
        
        self.mAuthorReviewPerComment=[]
        
        self.mCommentPhrases=[]
        
        productCount={}
        self.productAvgStars={}
        for rawReview in self.mRawCsvReviews:
            if(rawReview["Product"] not in self.productAvgStars.keys()):
                self.productAvgStars[rawReview["Product"]]=float(rawReview["Star Rating"])
                productCount[rawReview["Product"]]=1
            else:
                self.productAvgStars[rawReview["Product"]]+=float(rawReview["Star Rating"])
                productCount[rawReview["Product"]]+=1
            
        for key in self.productAvgStars.keys():
            self.productAvgStars[key]=float(self.productAvgStars[key])/float(productCount[key])
               
        self.mReviewAuthorMap={}
        self.mReviewStarMap={}
        for rawReview in self.mRawCsvReviews:
            self.mReviewAuthorMap[rawReview["Review_ID"]]=rawReview["Author"]
            self.mReviewStarMap[rawReview["Review_ID"]]=rawReview["Star Rating"]
            
        # Convert to lower case, remove punctuation, and assign parts of speech, etc...
        for itrComment, rawCsvCommentDict in enumerate( self.mRawCsvComments ):
            logging.getLogger("Context").info("Processing (1-gram) comment " + str(itrComment) + " of " + str(len(self.mRawCsvComments)) )
            
            
            # Extract review identifier
            reviewId = rawCsvCommentDict["Review_ID"]
            
            # Extract author of comment
            author = rawCsvCommentDict["Author"]
            
            if reviewId not in self.mAuthorFreqPerReview.keys():
                self.mAuthorFreqPerReview[reviewId]={}
                self.mAuthorFreqPerReview[reviewId][author]=1
            elif author not in self.mAuthorFreqPerReview[reviewId].keys():
                self.mAuthorFreqPerReview[reviewId][author]=1
            else:
                self.mAuthorFreqPerReview[reviewId][author]+=1
            
            self.mAuthorReviewPerComment.append((reviewId,author))
            
            # Append any unique review identifiers
            self.mReviewIds.update([reviewId])
            
            # Convert comment to lower case
            comment = rawCsvCommentDict["Comment"].lower();
            
            
            punctTokenizedComment = nltk.WordPunctTokenizer().tokenize(comment)
            
            phraseSeparators=['.','?','!',';']
            
            phrases=[]
            phrase=[]
            for word in punctTokenizedComment:
                if word in phraseSeparators:
                    phrase = [ phraseWord for phraseWord in phrase if (phraseWord not in self.mStopWords)]
                    phrase = nltk.pos_tag(phrase)
                    phrase = [(stemmer.stem(word),part) for (word,part) in phrase]
                    phrases.append(phrase)
                    phrase=[]
                else:
                    phrase.append(word)
                    
            if len(phrase)>0:    
                phrase = [ phraseWord for phraseWord in phrase if (phraseWord not in self.mStopWords)]
                phrase = nltk.pos_tag(phrase)
                phrases.append(phrase)
                    
            self.mCommentPhrases.append(phrases)
     
            # Replace punctuation with white space
            for punct in string.punctuation:
                comment=comment.replace(punct," ")
            
            self.mLowerCasePunctRemovedComments.append( comment )
            
            # Tokenize into list of words
            tokenizedComment = nltk.word_tokenize( comment )
            
                    
            # Filter out stop words
            tokenizedComment[:] = [ word for word in tokenizedComment if ( word not in self.mStopWords ) ]     
        
            
            
            posTagComment=nltk.pos_tag(tokenizedComment)
            # Append a list of (word, part of speech) tuples
            self.mPartOfSpeechTokenizedComments.append( posTagComment)
            
            self.mPartOfSpeechTokenizedCommentsAndReviewId.append((posTagComment,reviewId))
            
            # Append a list of stemmed words
            self.mStemmedTokenizedComments.append( [] )
            self.mStemmedTokenizedComments[-1][:] = [ stemmer.stem(word) for word in tokenizedComment ]
            
            # Assert parallel lists are same length
            assert( len( self.mPartOfSpeechTokenizedComments[-1] ) == len( self.mStemmedTokenizedComments[-1] ) )
                
            # Determine word counts for nouns and adjectives
            for itr, (word, partOfSpeech) in enumerate( self.mPartOfSpeechTokenizedComments[-1] ):
                # Determine if part of speech is noun or adjective
                if ( MinerMiscUtils.isAdj( partOfSpeech ) or MinerMiscUtils.isNoun( partOfSpeech ) ):
                    # Obtain stemmed word
                    stemmedWord = self.mStemmedTokenizedComments[-1][ itr ]
                    # Increment stemmed word counts
                    if ( stemmedWord in self.mAdjAndNounWordCountMap):
                        self.mAdjAndNounWordCountMap[ stemmedWord ] += 1
                        self.mStemmedWordToReviewsMap[ stemmedWord ].update( [ reviewId ] )
                    else:
                        self.mAdjAndNounWordCountMap[ stemmedWord ] = 1
                        self.mStemmedWordToReviewsMap[ stemmedWord ] = set( [ reviewId ] )
            # end inner for loop : iteration of (word, part of speech) tuples in single comment
        # end outer for loop : iteration over raw csv comment data
        
        # Assert parallel arrays are same length
        assert( len( self.mRawCsvComments ) == len( self.mLowerCasePunctRemovedComments ) )
        assert( len( self.mLowerCasePunctRemovedComments ) == len( self.mPartOfSpeechTokenizedComments ) )
        assert( len( self.mPartOfSpeechTokenizedComments ) == len( self.mStemmedTokenizedComments ) )
    
        # Set of words filtered by word counts: extract only words between threshold count ranges
        fGetWordReviewOverlap = lambda stemmedWord : float( len ( self.mStemmedWordToReviewsMap[ stemmedWord ] ) ) / float( len( self.mReviewIds ) ) 
        self.mFilteredWords = [ (word,count) for (word,count) in self.mAdjAndNounWordCountMap.iteritems() if  ( fGetWordReviewOverlap( word ) > filterWordReviewOverlap ) ]
        
        #Use the resulting filtered words as possible components of a phrase
        self.mPossiblePhraseWords = [word for (word,count) in self.mFilteredWords]
        
        #Count of 2-gram occurences
        self.mTwoGramsCountMap = {}
        
        #Count of the number of reviews the 2-grams occur in
        self.mTwoGramsToReviewsMap = {}
        
        #Extract all 2-grams from the comments
        for itrComment, (tokComment,reviewId) in enumerate( self.mPartOfSpeechTokenizedCommentsAndReviewId ):
            logging.getLogger("Context").info("Processing (2-grams) comment " + str(itrComment) + " of " + str(len(self.mRawCsvComments)) )
            
            #Keeps track of the previous word scanned
            prevWord="$"
            for itr, (word, partOfSpeech) in enumerate( tokComment ):
                
                # Determine if part of speech is noun or adjective
                if ( MinerMiscUtils.isAdj( partOfSpeech ) or MinerMiscUtils.isNoun( partOfSpeech ) ):
                    
                    # Obtain stemmed word
                    stemmedWord = stemmer.stem(word)
                    
                    # Increment stemmed 2-gram counts
                    if ( stemmedWord in self.mPossiblePhraseWords or prevWord in self.mPossiblePhraseWords):
                        phrase1 = prevWord + " " + stemmedWord
                        phrase2 = stemmedWord + " " + prevWord
                        defaultPhrase = phrase1
                        
                        if ( phrase2 in self.mTwoGramsCountMap.keys() ):
                            defaultPhrase = phrase2
                            self.mTwoGramsCountMap[defaultPhrase]+=1
                            self.mTwoGramsToReviewsMap[defaultPhrase].update(set(reviewId))
                        elif ( phrase1 in self.mTwoGramsCountMap.keys() ):
                            self.mTwoGramsCountMap[defaultPhrase]+=1
                            self.mTwoGramsToReviewsMap[defaultPhrase].update(set(reviewId))
                        else:
                            self.mTwoGramsCountMap[defaultPhrase]=1
                            self.mTwoGramsToReviewsMap[defaultPhrase]=set(reviewId)
                        
                    prevWord=stemmedWord
        
        #Extract all 2-grams that occur frequently enough across reviews to care about and add them to the set of "filtered words"
        #TODO: There should really be a separate collection for 2-grams
        for twoGram in self.mTwoGramsCountMap.keys():
            if(float(len(self.mTwoGramsToReviewsMap[twoGram])))/float( len( self.mReviewIds ))>(filterWordReviewOverlap*filterWordReviewOverlap):
                self.mFilteredWords.append((twoGram,self.mTwoGramsCountMap[twoGram]))
        
        self.printFilteredWords()

    # Print filtered words
    def printFilteredWords(self):
        for itr, (word, count) in enumerate( self.mFilteredWords ):
            logging.getLogger("Context").info( str(itr) + ": " + word + " (" + str(count) + ")" )
    