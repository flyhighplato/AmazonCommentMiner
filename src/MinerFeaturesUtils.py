'''
MinerFeaturesUtils.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerMiscUtils
import nltk
import string
import copy
import sys
import math

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
            
        outFeaturesMaps[ itrComment ][reviewStarsKey]=ctx.mReviewStarMap[reviewId]
        
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

# Structure for representing a CAR
class CAR:
    # Constructor
    def __init__( self, condSet=[], label=-1, condSetCount=0.0, labelCount=0.0, support=0.0, confidence=0.0 ):
        self.condSet = condSet
        self.label = label
        self.condSetCount = condSetCount
        self.labelCount = labelCount
        self.support = support
        self.confidence = confidence

    # String representation
    def __repr__(self):
        return "<" + str(self.condSet) + ", label=" + str(self.label) + ", cSup=" + str(self.condSetCount) + ", rSup=" + str(self.labelCount) + ", sup=" + str(self.support) + ", conf=" + str(self.confidence) + ">"
    
    # Returns true if condSet is contained in this features map, false otherwise
    def isContained(self, commentFeaturesMap, flattenedFeaturesMap ):
        for flattenedFeature in self.condSet:
            assert( flattenedFeature in flattenedFeaturesMap )
            expandedFeature = flattenedFeaturesMap[ flattenedFeature ]
            featureKey = expandedFeature[0]
            featureVal = expandedFeature[1]
            if ( featureKey not in commentFeaturesMap ):
                return False
            elif ( commentFeaturesMap[ featureKey ] != featureVal ):
                return False
        return True
    
# initializes labels for each comment and returns the set of unique labels
def CAR_get_comment_labels( ctx ):
    logging.getLogger( "CAR" ).info( "get comment labels" )
    outCommentLabels = []
    for rawCsvCommentDict in ctx.mRawCsvComments:
        outCommentLabels.append( MinerMiscUtils.getCommentLabel(rawCsvCommentDict) )
    return outCommentLabels

# @return support counts map by label of dictionaries of candidate features
def CAR_init_pass( featuresMaps, commentLabels, outCandidateCARObjs ) :
    logging.getLogger("CAR").info( "init pass" )
    
    # Maps a flattened feature key to its expanded (key, value) pair
    outFlattenedFeaturesMap = {}

    # A map of all unique candidate cars and their counts
    uniqueCARsMap = {}

    # A counter and a map to enumerate the features in an attempt to avoid string comparisons
    uniqueFeaturesCounter = 0
    uniqueFeaturesEnumerationMap = {}
    
    # Find all unique candidate CARs
    for itrComment, commentFeaturesMap in enumerate(featuresMaps):
        label = commentLabels[ itrComment ]
        for featureKey, featureValue in commentFeaturesMap.iteritems():
            # Merge feature key and value into a single key
            flattenedFeature = str(featureKey) + "=" + str(featureValue)
            # Merge label onto flattened feature
            uniqueCARKey = flattenedFeature + ":" + str(label)
            if uniqueCARKey not in uniqueCARsMap:
                # Store CAR feature and label
                if flattenedFeature not in uniqueFeaturesEnumerationMap:
                    # Update mapping to expanded feature key, value pairs
                    uniqueFeaturesEnumerationMap[ flattenedFeature ] = uniqueFeaturesCounter
                    outFlattenedFeaturesMap[ uniqueFeaturesCounter ] = [ featureKey, featureValue ]
                    uniqueFeaturesCounter += 1
                # Map enumerated feature to CAR object
                uniqueCARsMap[ uniqueCARKey ] = CAR( [uniqueFeaturesEnumerationMap[flattenedFeature]], label )
    
    # Transform into a list of CAR objects
    outCandidateCARObjs[:] = [ uniqueCARsMap[ uniqueCARKey ] for uniqueCARKey in uniqueCARsMap ]
    
    # Return resulting features map
    return outFlattenedFeaturesMap

def CAR_update_candidate_counts( featuresMaps, commentLabels, flattenedFeaturesMap, outCandidateCARObjs ):
    logging.getLogger("CAR").info( "update candidate counts" )
    for itrComment, commentFeaturesMap in enumerate(featuresMaps):
        label = commentLabels[ itrComment ]
        for CARObj in outCandidateCARObjs:
            if CARObj.isContained( commentFeaturesMap, flattenedFeaturesMap ):
                CARObj.condSetCount += 1.0
                if label == CARObj.label:
                    CARObj.labelCount += 1.0
    
def CAR_extract_frequent_rules( candidateCARObjs, minSup, minConf, n, outFrequentCARObjs ):
    logging.getLogger("CAR").info( "extract frequent rules" )
    assert( n > 0.0 )
    #minRecordedSupport = sys.maxint
    #minRecordedConfidence = sys.maxint
    #maxRecordedSupport = -1
    #maxRecordedConfidence = -1
    #avgRecordedSupport = 0
    #avgRecordedConfidence = 0
    outFrequentCARObjs[:] = []
    for CARObj in candidateCARObjs:
        CARObj.support = CARObj.labelCount / n
        assert( CARObj.condSetCount > 0.0 )
        CARObj.confidence = CARObj.labelCount / CARObj.condSetCount
        if ((CARObj.support >= minSup) and (CARObj.confidence >= minConf)):
            outFrequentCARObjs.append( CARObj )
        # Update stats for support
        #if ( CARObj.support < minRecordedSupport ):
        #    minRecordedSupport = CARObj.support
        #elif ( CARObj.support > maxRecordedSupport ):
        #    maxRecordedSupport = CARObj.support
        #avgRecordedSupport += CARObj.support
        # Update stats for confidence
        #if ( CARObj.confidence < minRecordedConfidence ):
        #    minRecordedConfidence = CARObj.confidence
        #elif ( CARObj.confidence > maxRecordedConfidence ):
        #    maxRecordedConfidence = CARObj.confidence
        #avgRecordedConfidence += CARObj.confidence
    # Dump stats
    #avgRecordedSupport = avgRecordedSupport / len( candidateCARObjs )
    #avgRecordedConfidence = avgRecordedConfidence / len( candidateCARObjs )
    #logging.getLogger("CAR").info( "\tsupRange=["+str(minRecordedSupport)+","+str(maxRecordedSupport)+"]" )
    #logging.getLogger("CAR").info( "\tconfRange=["+str(minRecordedConfidence)+","+str(maxRecordedConfidence)+"]" )
    #logging.getLogger("CAR").info( "\tavgSup="+str(avgRecordedSupport)+", avgConf="+str(avgRecordedConfidence) )
    # Determine variance and std deviation
    #varRecordedSupport = 0
    #varRecordedConfidence = 0
    #for CARObj in candidateCARObjs:
    #    sqDistSupport = CARObj.support - avgRecordedSupport
    #    sqDistSupport *= sqDistSupport
    #    varRecordedSupport += sqDistSupport
    #    sqDistConfidence = CARObj.confidence - avgRecordedConfidence
    #    sqDistConfidence *= sqDistConfidence
    #    varRecordedConfidence += sqDistConfidence
    #varRecordedSupport /= len( candidateCARObjs )
    #varRecordedConfidence /= len( candidateCARObjs )
    #logging.getLogger("CAR").info( "\tvarSup="+str(varRecordedSupport)+", stddevSup="+str(math.sqrt(varRecordedSupport)) )
    #logging.getLogger("CAR").info( "\tvarConf="+str(varRecordedConfidence)+", stddevConf="+str(math.sqrt(varRecordedConfidence)) )

def CAR_candidate_gen( prevFrequentCARObjs, outCandidateCARObjs ):
    logging.getLogger("CAR").info( "candidate gen" )
    outCandidateCARObjs[:] = []
    for idxA in range( len(prevFrequentCARObjs) ):
        CARObjA = prevFrequentCARObjs[ idxA ]
        logging.getLogger("CAR").info( "Processing " + str(len(CARObjA.condSet)) + "-CAR " + str(idxA+1) + " of " + str(len(prevFrequentCARObjs)) )
        # Skip k-1 rules that are ~100% confidence
        if ( CARObjA.confidence >= 0.99 ):
            continue
        for idxB in range( idxA+1, len(prevFrequentCARObjs) ):
            CARObjB = prevFrequentCARObjs[ idxB ]
            assert( len( CARObjA.condSet ) == len( CARObjB.condSet ) )
            # Skip CARs that don't have same class
            if ( CARObjA.label != CARObjB.label ):
                continue
            # Skip k-1 rules that are ~100% confidence
            if ( CARObjB.confidence >= 0.99 ):
                continue
            # Skip CARs that do not differ in the last item only
            bCanJoin = True
            for idxFeature in range( len( CARObjA.condSet ) - 1 ):
                if ( CARObjA.condSet[ idxFeature ] != CARObjB.condSet[ idxFeature ] ):
                    bCanJoin = False
                    break
            if ( bCanJoin == False ):
                continue
            # Join both CARs while maintaining lexographic order
            joinedCARObj = CAR( [], CARObjA.label )
            assert( CARObjA.condSet[-1] != CARObjB.condSet[-1] )
            if ( CARObjA.condSet[-1] < CARObjB.condSet[-1]):
                joinedCARObj.condSet = copy.deepcopy( CARObjA.condSet )
                joinedCARObj.condSet.append( CARObjB.condSet[-1] )
            else:
                joinedCARObj.condSet = copy.deepcopy( CARObjB.condSet )
                joinedCARObj.condSet.append( CARObjA.condSet[-1] )
            # Determine if joined CAR should be pruned
            bShouldPrune = False
            for idxFeatureToDrop in range( len(joinedCARObj.condSet) ):
                subCondSet = copy.deepcopy( joinedCARObj.condSet )
                subCondSet.pop( idxFeatureToDrop )
                bSubCondSetFound = False
                for CARObjForPruning in prevFrequentCARObjs:
                    if ( CARObjForPruning.label != joinedCARObj.label ):
                        continue
                    if ( CARObjForPruning.condSet == subCondSet ):
                        bSubCondSetFound = True
                        break
                if ( bSubCondSetFound == False ):
                    bShouldPrune = True
                    break
            if ( bShouldPrune == False ):
                outCandidateCARObjs.append( joinedCARObj )
    
def addFeaturesCAR( ctx, outFeaturesMaps ):
    minSup = 0.1
    minConf = 0.5
    n = len(outFeaturesMaps)
    logging.getLogger("Features").info( "CAR minSup="+str(minSup)+" minConf="+str(minConf)+" n="+str(n) )

    # Get labels for each comment
    commentLabels = CAR_get_comment_labels( ctx )
    
    # History of candidate sequences
    CHist = [[]]
    FHist = [[]]
    flattenedFeaturesMap = CAR_init_pass( outFeaturesMaps, commentLabels, CHist[0] )
    CAR_update_candidate_counts( outFeaturesMaps, commentLabels, flattenedFeaturesMap, CHist[0] )
    CAR_extract_frequent_rules( CHist[0], minSup, minConf, n, FHist[0] )
    logging.getLogger("CAR").info( str(len(CHist[-1])) + " Candidate 1-sequences have been generated.")
    logging.getLogger("CAR").info( str(len(FHist[-1])) + " Frequent 1-sequences have been generated.")
    
    maxK = 10
    for idxK in range( 1, maxK ):
        if ( len( FHist[-1] ) <= 0 ):
            break
        assert( len( CHist) == idxK )
        assert( len( FHist) == idxK )
        CHist.append( [] )
        CAR_candidate_gen( FHist[-1], CHist[-1] )
        CAR_update_candidate_counts( outFeaturesMaps, commentLabels, flattenedFeaturesMap, CHist[-1] )
        logging.getLogger("CAR").info( str(len(CHist[-1])) + " Candidate " + str(idxK+1) + "-sequences have been generated.")
        FHist.append( [] )
        CAR_extract_frequent_rules( CHist[-1], minSup, minConf, n, FHist[-1] )
        logging.getLogger("CAR").info( str(len(FHist[-1])) + " Frequent " + str(idxK+1) + "-sequences have been generated.")
        
    #logging.getLogger("CAR").info( "CHist" + str(CHist) )
    #logging.getLogger("CAR").info( "FHist" + str(FHist) )

    