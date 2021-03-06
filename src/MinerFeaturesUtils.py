'''
MinerFeaturesUtils.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerMiscUtils
import math
import nltk
import string
import MinerCAR
import pickle

# Function callback offsets 
class eFeaturesMaskBits:
    wordExists = 1 << 0
    commentLength = 1 << 1
    helpfullnessRatio = 1 << 2
    authorFreqInReview = 1 << 3
    reviewAuthorMentioned = 1 << 4
    commentAuthorMentioned = 1 << 5
    dist = 1 << 6
    phrases = 1 << 7
    CAR = 1 << 8

# Appends an empty features set dictionary for each comment
def initFeatures( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "init" )
    for comment in ctx.mRawCsvComments:
        outFeaturesMaps.append( {} )

def addFeaturesCommentLength( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "comment length" )
    lengthKey = "L_"
    for itr, comment in enumerate( ctx.mLowerCasePunctRemovedComments ):
        if ( len( comment ) > 100 ):
            outFeaturesMaps[ itr ][ lengthKey ] =  1 # High
        else:
            outFeaturesMaps[ itr ][ lengthKey ] = -1 # Low

def addFeaturesHelpfulnessRatio( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "helpfulness ratio" )
    helpfullnessRatioKey = "HR"
    for itrComment, rawCsvCommentDict in enumerate( ctx.mRawCsvComments ):
        hr = float( rawCsvCommentDict[ "Helpfullness Ratio" ] )       
        if ( hr > 0 and hr < 0.2 ):
            outFeaturesMaps[ itrComment ][ helpfullnessRatioKey ] = -2 # "LOW"
        elif ( hr >= 0.2 and hr < 0.5 ):
            outFeaturesMaps[ itrComment ][ helpfullnessRatioKey ] = -1 # "MLOW"
        elif ( hr >= 0.5 and hr < 0.8 ):
            outFeaturesMaps[ itrComment ][ helpfullnessRatioKey ] =  1 # "MHIGH"
        else: # ( hr < 1 and hr >= 0.8 ):
            outFeaturesMaps[ itrComment ][ helpfullnessRatioKey ] =  2 # "HIGH"

def addFeaturesPhrases( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "phrases" )
    rawFilteredWords = [ word for ( word, count ) in ctx.mFilteredWords ]
    for itrComment, phrases in enumerate(ctx.mCommentPhrases):
        for phrase in phrases:
            prevWord='$'
            for itrWord, (word, partOfSpeech) in enumerate( phrase ):
                if ( MinerMiscUtils.isAdj( partOfSpeech ) or MinerMiscUtils.isNoun( partOfSpeech ) ):
                    stemmedWord = word
                    
                    phrase1 = prevWord + " " + stemmedWord
                    phrase2 = stemmedWord + " " + prevWord
                    
                    if(phrase1 in rawFilteredWords):
                        #print "Adding:" + phrase1
                        outFeaturesMaps[ itrComment ][ phrase1 ] = 1
                    elif(phrase2 in rawFilteredWords):
                        #print "Adding:" + phrase1
                        outFeaturesMaps[ itrComment ][ phrase2 ] = 1  
                   
                    prevWord = stemmedWord
                
def addFeaturesWordExists( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "word exists" )
    for itrComment, stemmedTokenizedComment in enumerate( ctx.mStemmedTokenizedComments ):
        for ( word, count ) in ctx.mAdjAndNounWordCountMap.iteritems():
        #for ( word, count ) in ctx.mFilteredWords: 
            if ( stemmedTokenizedComment.count( word ) > 0 ):
                outFeaturesMaps[ itrComment ][ word ] = 1
            else:
                outFeaturesMaps[ itrComment ][ word ] = 0

def addFeaturesAuthorFreqInReview( ctx, outFeaturesMaps):
    logging.getLogger("Features").info( "author frequency" )
    multiCommentKey="M-C"
    isAuthorKey="I-A"
    reviewStarsKey="R-S"
    reviewStarsDeviationKey = "R-SD"
    for itrComment, (reviewId,author) in enumerate( ctx.mAuthorReviewPerComment ):
        if(ctx.mAuthorFreqPerReview[reviewId][author]>5):
            outFeaturesMaps[ itrComment ][multiCommentKey]=2
        elif(ctx.mAuthorFreqPerReview[reviewId][author]>1):
            outFeaturesMaps[ itrComment ][multiCommentKey]=1
        else:
            outFeaturesMaps[ itrComment ][multiCommentKey]=0
        if(ctx.mReviewAuthorMap[reviewId]==author):
            outFeaturesMaps[ itrComment ][isAuthorKey]=1
        else:
            outFeaturesMaps[ itrComment ][isAuthorKey]=0
            
        outFeaturesMaps[ itrComment ][reviewStarsKey]=float(ctx.mReviewStarMap[reviewId])
        
        if(ctx.mReviewStarMap[reviewId]>ctx.productAvgStars):
            outFeaturesMaps[ itrComment ][reviewStarsDeviationKey]=-1
        elif(ctx.mReviewStarMap[reviewId]>ctx.productAvgStars):
            outFeaturesMaps[ itrComment ][reviewStarsDeviationKey]=1
        else:
            outFeaturesMaps[ itrComment ][reviewStarsDeviationKey]=0

def isNameInComment( ctx, name, itrComment ):
    name = name.lower()
    
    # Filter out punctuation from author name
    for punct in string.punctuation:
            name=name.replace(punct," ")
     
    bNameInComment = 0
    if ( -1 != ctx.mLowerCasePunctRemovedComments[ itrComment ].find( name ) ):
        bNameInComment = 1
     
    # Tokenize author name and search for tokens
    if ( 0 == bNameInComment ):
        tokenizedName = []
        for nameToken in nltk.word_tokenize( name ):
            if ( ( len( nameToken ) > 2 ) and ( nameToken not in ctx.mStopWords ) ):
                tokenizedName.append( nameToken )
        
        # Check to see if a token exists in the comment
        for nameToken in tokenizedName:
            if ( -1 != ctx.mLowerCasePunctRemovedComments[ itrComment ].find( nameToken ) ):
                    bNameInComment = 1
                    break
    # return 1 if name was found, 0 otherwise
    return bNameInComment

def addFeaturesReviewAuthorMentioned( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "Review Author Mentioned" )
    assert( len( ctx.mAuthorReviewPerComment ) == len( ctx.mLowerCasePunctRemovedComments ) )
    reviewAuthorNameInCommentKey="RANC-"
    for itrComment, (reviewId,authorOfComment) in enumerate( ctx.mAuthorReviewPerComment ):
        assert( reviewId in ctx.mReviewAuthorMap )
        # Convert author of review to lowercase
        authorOfReview = ctx.mReviewAuthorMap[ reviewId ]
        outFeaturesMaps[ itrComment ][ reviewAuthorNameInCommentKey ] = isNameInComment( ctx, authorOfReview, itrComment )

def addFeaturesCommentAuthorMentioned( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "Comment Author Mentioned" )
    
    # Create map from review id to list of comment authors
    reviewIdToCommentAuthorListMap = {}
    for (reviewId,commentAuthor) in ctx.mAuthorReviewPerComment:
        if ( reviewId not in reviewIdToCommentAuthorListMap ):
            reviewIdToCommentAuthorListMap[ reviewId ] = []
        if ( commentAuthor == ctx.mReviewAuthorMap[ reviewId ] ):
            continue
        
        if ( commentAuthor not in reviewIdToCommentAuthorListMap[reviewId] ):
                reviewIdToCommentAuthorListMap[ reviewId ].append( commentAuthor )
    
    commentAuthorNameInCommentKey = "CANC-"
    for itrComment, (reviewId,authorOfComment) in enumerate( ctx.mAuthorReviewPerComment ):
        bNameFound = 0
        for name in reviewIdToCommentAuthorListMap[ reviewId ]:
            if ( isNameInComment( ctx, name, itrComment ) ):
                bNameFound = 1
                break
        outFeaturesMaps[ itrComment ][ commentAuthorNameInCommentKey ] = bNameFound

def addFeaturesDist( ctx, outFeaturesMaps ):
    logging.getLogger("Features").info( "Distance" )

    # Centroids for each class label
    centroids = [{}, {}]    
    centroidsCacheFileName = "centroidsCache.txt"
    if ( MinerMiscUtils.fileExists(centroidsCacheFileName)):
        # Load from cache!
        centroids = pickle.load( open( centroidsCacheFileName ) )
    else:
        # Sum up all features vectors
        for itrComment, rawCsvCommentDict in enumerate( ctx.mRawCsvComments ):
            label = MinerMiscUtils.getCommentLabel(rawCsvCommentDict)
            for key, value in outFeaturesMaps[ itrComment ].iteritems():
                if ( type( value ) is str ):
                    print "BREAK = " + key + " = " + value + "\n"
                if ( key in centroids[label]):
                    centroids[label][key] += value
                else:
                    centroids[label][key] = value
                
                for altLabel in range( len(centroids ) ):
                    if ( altLabel != label ):
                        if key not in centroids[ altLabel ]:
                            centroids[ altLabel ][key] = 0.0
        
        # Average the centroids
        for centroid in centroids:
            for key, value in centroid.iteritems():
                value /= len( outFeaturesMaps )
                centroid[ key ] = value
        
        # Cache the centroids to disk
        pickle.dump( centroids, open( centroidsCacheFileName, "wb" ) )
        
    # Determine distance from both centroids
    distances = [ [], [] ]
    averageDistance = [ 0.0, 0.0 ]    
    for featuresMap in outFeaturesMaps:
        for label, centroid in enumerate(centroids):
            totalSqDist = 0.0
            for centroidKey, centroidValue in centroid.iteritems():
                commentValue = 0.0
                if ( centroidKey in featuresMap ):
                    commentValue = featuresMap[ centroidKey ]
                sqDist = commentValue - centroidValue
                sqDist *= sqDist
                totalSqDist += sqDist
            totalDist = math.sqrt( totalSqDist )
            distances[label].append( totalDist )
            averageDistance[label] += totalDist 
    
    for label in range( len( averageDistance ) ):
        averageDistance[ label ] /= len( outFeaturesMaps )
    
    # Determine standard deviation
    averageStdDev = [ 0, 0 ]
    for label, labelDistances in enumerate( distances ):
        for distance in labelDistances:
            sqDistFromMean = distance - averageDistance[ label ]
            sqDistFromMean *= sqDistFromMean
            averageStdDev[ label ] += sqDistFromMean
    
    for label in range( len( averageStdDev ) ):
        averageStdDev[ label ] /= len( outFeaturesMaps )
        averageStdDev[ label ]  = math.sqrt( averageStdDev[ label ] )
        
    # Map all feature vectors as being closer or farther from std dev
    for itrComment, featuresMap in enumerate(outFeaturesMaps):
        for label, stdDev in enumerate( averageStdDev ):
            featuresMap[ "Dist--"+str(label) ] = distances[ label ][ itrComment ] > stdDev
    
def addFeaturesCAR( ctx, outFeaturesMaps, minSup=0.1, minConf=0.6, cacheFileName="CARcache.txt" ):
    logging.getLogger("Features").info( "CAR" )
    # Note: average support is around 0.28, stddev is ~0.2 (i think)
    # Note: average confidence is around 0.58, stddev is ~0.2
    FHistFlattenedFeaturesMapPair = MinerCAR.CAR_conditional_apriori( ctx, outFeaturesMaps, cacheFileName, minSup, minConf )
   
    FHist = FHistFlattenedFeaturesMapPair[0]
    flattenedFeaturesMap = FHistFlattenedFeaturesMapPair[1]
    uniqueCarKeysMap = {}
   
    for itrLevel in range( 1, len(FHist) ):
        for CARObj in FHist[itrLevel]:
            CARKey = ""
            for enumFeatureKey in CARObj.condSet:
                CARKey += str(flattenedFeaturesMap[enumFeatureKey][0])+"="+str(flattenedFeaturesMap[enumFeatureKey][1])+","
            if ( CARKey in uniqueCarKeysMap ):
                continue
            logging.getLogger("Features").info( "Adding "+str(len(CARObj.condSet))+"-CAR: " + str(CARKey) )
            for commentFeaturesMap in outFeaturesMaps:
                commentFeaturesMap[ CARKey ] = CARObj.isContained( commentFeaturesMap, flattenedFeaturesMap )

