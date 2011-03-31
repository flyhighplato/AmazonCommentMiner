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
    def __init__( self, strPathToRawCsvComments, filterWordCountMin, filterWordCountMax, filterWordReviewOverlap ):
        # Log parameters
        logging.getLogger("Context").info( "Creating new context:" )
        logging.getLogger("Context").info( "strPathToRawCsvComments: " + strPathToRawCsvComments )
        logging.getLogger("Context").info( "filterWordCountMin: " + str(filterWordCountMin) )
        logging.getLogger("Context").info( "filterWordCountMax: " + str(filterWordCountMax) )
        logging.getLogger("Context").info( "filterWordReviewOverlap: " + str(filterWordReviewOverlap) )
        
        # Load stop words
        self.mStopWords = nltk.corpus.stopwords.words('english')
                                                      
        # Load and shuffle comment data
        self.mRawCsvComments = csv.DictReader(open(strPathToRawCsvComments))
        self.mRawCsvComments = [comment for comment in self.mRawCsvComments]
        random.shuffle(self.mRawCsvComments)
        # self.mRawCsvComments=self.mRawCsvComments[0:100]
        
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
        
        # Convert to lower case, remove punctuation, and assign parts of speech, etc...
        for itrComment, rawCsvCommentDict in enumerate( self.mRawCsvComments ):
            logging.getLogger("Context").info("Processing comment " + str(itrComment) + " of " + str(len(self.mRawCsvComments)) )
            
            # Extract review identifier
            reviewId = rawCsvCommentDict["Review_ID"]
            
            # Append any unique review identifiers
            self.mReviewIds.update([reviewId])
            
            # Convert comment to lower case
            comment = rawCsvCommentDict["Comment"].lower();
     
            # Replace punctuation with white space
            for punct in string.punctuation:
                comment=comment.replace(punct," ")
            
            self.mLowerCasePunctRemovedComments.append( comment )
            
            # Tokenize into list of words
            tokenizedComment = nltk.word_tokenize( comment )
            
            # Filter out stop words
            tokenizedComment[:] = [ word for word in tokenizedComment if ( word not in self.mStopWords ) ]     
        
            # Append a list of (word, part of speech) tuples
            self.mPartOfSpeechTokenizedComments.append( nltk.pos_tag(tokenizedComment) )
            
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
        
        self.printFilteredWords()

    # Print filtered words
    def printFilteredWords(self):
        for itr, (word, count) in enumerate( self.mFilteredWords ):
            logging.getLogger("Context").info( str(itr) + ": " + word + " (" + str(count) + ")" )
    