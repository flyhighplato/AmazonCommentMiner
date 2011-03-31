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
                if ( stemmedWord in rawFilteredWords and prevWord in rawFilteredWords ):
                    phrase1 = prevWord + " " + stemmedWord
                    phrase2 = stemmedWord + " " + prevWord
                    defaultPhrase = phrase1
                    if ( phrase2 in outFeaturesMaps[ itrComment ].keys() ):
                        defaultPhrase = phrase2
                    outFeaturesMaps[ itrComment ][ defaultPhrase ] = 1
                prevWord = stemmedWord
                
def addFeaturesWordExists( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "word exists" )
    for itrComment, stemmedTokenizedComment in enumerate( ctx.mStemmedTokenizedComments ):
        for ( word, count ) in ctx.mAdjAndNounWordCountMap.iteritems():  
            if ( stemmedTokenizedComment.count( word ) > 0 ):
                outFeaturesMaps[ itrComment ][ word ] = 1
            else:
                outFeaturesMaps[ itrComment ][ word ] = 0

                    
                    
                
                
        
                