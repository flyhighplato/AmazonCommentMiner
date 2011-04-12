'''
MinerFeaturesUtils.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerMiscUtils

# Appends an empty features set dictionary for each comment
def initFeatures( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "init" )
    for comment in ctx.mRawCsvComments:
        outFeaturesMaps.append( {} )

def addFeaturesCommentLength( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "comment length" )
    for itr, comment in enumerate( ctx.mLowerCasePunctRemovedComments ):
        if ( len( comment ) > 100 ):
            outFeaturesMaps[ itr ][ "LENGTH" ] =  1 # High
        else:
            outFeaturesMaps[ itr ][ "LENGTH" ] = -1 # Low

def addFeaturesHelpfulnessRatio( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "helpfulness ratio" )
    for itrComment, rawCsvCommentDict in enumerate( ctx.mRawCsvComments ):
        hr = float( rawCsvCommentDict[ "Helpfullness Ratio" ] )       
        if ( hr > 0 and hr < 0.2 ):
            outFeaturesMaps[ itrComment ][ "HR" ] = -2 # "LOW"
        elif ( hr >= 0.2 and hr < 0.5 ):
            outFeaturesMaps[ itrComment ][ "HR" ] = -1 # "MLOW"
        elif ( hr >= 0.5 and hr < 0.8 ):
            outFeaturesMaps[ itrComment ][ "HR" ] =  1 # "MHIGH"
        else: # ( hr < 1 and hr >= 0.8 ):
            outFeaturesMaps[ itrComment ][ "HR" ] =  2 # "HIGH"

def addFeaturesPhrases( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "phrases" )
    rawFilteredWords = [ word for ( word, count ) in ctx.mFilteredWords ]
    for itrComment, partOfSpeechTokenizedComment in enumerate( ctx.mPartOfSpeechTokenizedComments ):
        prevWord='$'
        for itrWord, (word, partOfSpeech) in enumerate( partOfSpeechTokenizedComment ):
            if ( MinerMiscUtils.isAdj( partOfSpeech ) or MinerMiscUtils.isNoun( partOfSpeech ) ):
                stemmedWord = ctx.mStemmedTokenizedComments[ itrComment ][ itrWord ]
                
                phrase1 = prevWord + " " + stemmedWord
                phrase2 = stemmedWord + " " + prevWord
                
                if(phrase1 in rawFilteredWords):
                    outFeaturesMaps[ itrComment ][ phrase1 ] = 1
                elif(phrase2 in rawFilteredWords):
                    outFeaturesMaps[ itrComment ][ phrase2 ] = 1  
               
                prevWord = stemmedWord
                
                
def addFeaturesWordExists( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "word exists" )
    for itrComment, stemmedTokenizedComment in enumerate( ctx.mStemmedTokenizedComments ):
        for ( word, count ) in ctx.mAdjAndNounWordCountMap.iteritems():  
            if ( stemmedTokenizedComment.count( word ) > 0 ):
                outFeaturesMaps[ itrComment ][ word ] = 1
            else:
                outFeaturesMaps[ itrComment ][ word ] = 0
                
def addFeaturesAuthorFreqInReview( ctx, outFeaturesMaps):
    logging.getLogger("Features").info( "author frequency" )
    for itrComment, (reviewId,author) in enumerate( ctx.mAuthorReviewPerComment ):
        if(ctx.mAuthorFreqPerReview[reviewId][author]>5):
            outFeaturesMaps[ itrComment ]["MULTI-COMMENT"]=2
        elif(ctx.mAuthorFreqPerReview[reviewId][author]>1):
            outFeaturesMaps[ itrComment ]["MULTI-COMMENT"]=1
        else:
            outFeaturesMaps[ itrComment ]["MULTI-COMMENT"]=0
        if(ctx.mReviewAuthorMap[reviewId]==author):
            outFeaturesMaps[ itrComment ]["IS-AUTHOR"]=1
        else:
            outFeaturesMaps[ itrComment ]["IS-AUTHOR"]=0
            
        outFeaturesMaps[ itrComment ]["REVIEW-STARS"]=ctx.mReviewStarMap[reviewId]
        
        if(ctx.mReviewStarMap[reviewId]>ctx.productAvgStars):
            outFeaturesMaps[ itrComment ]["REVIEW-STARS-DEVIATION"]=-1
        elif(ctx.mReviewStarMap[reviewId]>ctx.productAvgStars):
            outFeaturesMaps[ itrComment ]["REVIEW-STARS-DEVIATION"]=1
        else:
            outFeaturesMaps[ itrComment ]["REVIEW-STARS-DEVIATION"]=0
        

                    
                    
                
                
        
                