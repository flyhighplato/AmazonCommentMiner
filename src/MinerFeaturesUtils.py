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
        prevPrevWord="$"
        for itrWord, (word, partOfSpeech) in enumerate( partOfSpeechTokenizedComment ):
            if ( MinerMiscUtils.isAdj( partOfSpeech ) or MinerMiscUtils.isNoun( partOfSpeech ) ):
                stemmedWord = ctx.mStemmedTokenizedComments[ itrComment ][ itrWord ]
                
                #Add 2-grams
                if ( stemmedWord in rawFilteredWords and prevWord in rawFilteredWords ):
                    phrase1 = prevWord + " " + stemmedWord
                    phrase2 = stemmedWord + " " + prevWord
                    defaultPhrase = phrase1
                    if ( phrase2 in outFeaturesMaps[ itrComment ].keys() ):
                        defaultPhrase = phrase2
                    outFeaturesMaps[ itrComment ][ defaultPhrase ] = 1
                
                #Add 2-grams
                if ( stemmedWord in rawFilteredWords and prevWord in rawFilteredWords and prevPrevWord in rawFilteredWords):
                    phrase1 = prevWord + " " + stemmedWord + " " + prevPrevWord
                    phrase2 = prevWord + " " + prevPrevWord + " " + stemmedWord
                    phrase3 = stemmedWord + " " + prevWord + " " + prevPrevWord
                    phrase4 = stemmedWord + " " + prevPrevWord + " " + prevWord
                    phrase5 = prevPrevWord + " " + prevWord + " " + stemmedWord
                    phrase6 = prevPrevWord + " " + stemmedWord + " " + prevWord
                    
                    defaultPhrase = phrase5
                    
                    if ( phrase1 in outFeaturesMaps[ itrComment ].keys() ):
                        defaultPhrase = phrase1
                    elif ( phrase2 in outFeaturesMaps[ itrComment ].keys() ):
                        defaultPhrase = phrase2
                    elif ( phrase3 in outFeaturesMaps[ itrComment ].keys() ):
                        defaultPhrase = phrase3
                    elif ( phrase4 in outFeaturesMaps[ itrComment ].keys() ):
                        defaultPhrase = phrase4
                    elif ( phrase6 in outFeaturesMaps[ itrComment ].keys() ):
                        defaultPhrase = phrase6
                    
                    outFeaturesMaps[ itrComment ][ defaultPhrase ] = 1
                   
                    
                
                prevPrevWord = prevWord
                prevWord = stemmedWord
                
                
def addFeaturesWordExists( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "word exists" )
    for itrComment, stemmedTokenizedComment in enumerate( ctx.mStemmedTokenizedComments ):
        for ( word, count ) in ctx.mAdjAndNounWordCountMap.iteritems():  
            if ( stemmedTokenizedComment.count( word ) > 0 ):
                outFeaturesMaps[ itrComment ][ word ] = 1
            else:
                outFeaturesMaps[ itrComment ][ word ] = 0

                    
                    
                
                
        
                