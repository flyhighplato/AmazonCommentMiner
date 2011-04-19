'''
MinerCAR.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerMiscUtils
import copy
import pickle
import os

# Uncomment if want statistics
import sys
import math

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

class CARStats:
# Constructor
    def __init__( self ):
        self.minRecordedSupport = sys.maxint
        self.minRecordedConfidence = sys.maxint
        self.maxRecordedSupport = -1
        self.maxRecordedConfidence = -1
        self.avgRecordedSupport = 0
        self.avgRecordedConfidence = 0
        self.varRecordedSupport = 0
        self.varRecordedConfidence = 0
    
    # String representation
    def __repr__(self):
        strRepr =  "supRange=["+str(self.minRecordedSupport)+","+str(self.maxRecordedSupport)+"]\n"
        strRepr += "confRange=["+str(self.minRecordedConfidence)+","+str(self.maxRecordedConfidence)+"]\n"
        strRepr += "avgSup="+str(self.avgRecordedSupport)+", avgConf="+str(self.avgRecordedConfidence)+"\n"    
        strRepr += "varSup="+str(self.varRecordedSupport)+", stddevSup="+str(math.sqrt(self.varRecordedSupport))+"\n"
        strRepr += "varConf="+str(self.varRecordedConfidence)+", stddevConf="+str(math.sqrt(self.varRecordedConfidence))
        return strRepr
    
    def updateForCAR( self, CARObj ):
        # Update stats for support
        if ( CARObj.support < self.minRecordedSupport ):
            self.minRecordedSupport = CARObj.support
        elif ( CARObj.support > self.maxRecordedSupport ):
            self.maxRecordedSupport = CARObj.support
        self.avgRecordedSupport += CARObj.support
        # Update stats for confidence
        if ( CARObj.confidence < self.minRecordedConfidence ):
            self.minRecordedConfidence = CARObj.confidence
        elif ( CARObj.confidence > self.maxRecordedConfidence ):
            self.maxRecordedConfidence = CARObj.confidence
        self.avgRecordedConfidence += CARObj.confidence
    
    def finalize( self, candidateCARObjs ):
        self.avgRecordedSupport = self.avgRecordedSupport / len( candidateCARObjs )
        self.avgRecordedConfidence = self.avgRecordedConfidence / len( candidateCARObjs )
        # Determine variance and std deviation
        for CARObj in candidateCARObjs:
            sqDistSupport = CARObj.support - self.avgRecordedSupport
            sqDistSupport *= sqDistSupport
            self.varRecordedSupport += sqDistSupport
            sqDistConfidence = CARObj.confidence - self.avgRecordedConfidence
            sqDistConfidence *= sqDistConfidence
            self.varRecordedConfidence += sqDistConfidence
        self.varRecordedSupport /= len( candidateCARObjs )
        self.varRecordedConfidence /= len( candidateCARObjs )

def CAR_extract_frequent_rules( candidateCARObjs, minSup, minConf, n, outFrequentCARObjs ):
    logging.getLogger("CAR").info( "extract frequent rules" )
    assert( n > 0.0 )
    stats = CARStats()
    outFrequentCARObjs[:] = []
    for CARObj in candidateCARObjs:
        CARObj.support = CARObj.labelCount / n
        if ( CARObj.condSetCount > 0.001 ):
            CARObj.confidence = CARObj.labelCount / CARObj.condSetCount
        else:
            CARObj.confidence = 0.0
        if ((CARObj.support >= minSup) and (CARObj.confidence >= minConf)):
            outFrequentCARObjs.append( CARObj )
        # Update stats for support
        stats.updateForCAR(CARObj)
    stats.finalize( candidateCARObjs )
    logging.getLogger("CAR").info( "stats=" + str(stats) )

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
    
def CAR_apriori( ctx, featuresMaps, cacheFileName, minSup=0.1, minConf=0.5 ):
    n = len(featuresMaps)
    logging.getLogger("CAR").info( "apriori minSup="+str(minSup)+" minConf="+str(minConf)+" n="+str(n) )

    # Get labels for each comment
    commentLabels = CAR_get_comment_labels( ctx )
    
    # History of candidate sequences
    CHist = [[]]
    FHist = [[]]
    flattenedFeaturesMap = CAR_init_pass( featuresMaps, commentLabels, CHist[0] )
    CAR_update_candidate_counts( featuresMaps, commentLabels, flattenedFeaturesMap, CHist[0] )
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
        CAR_update_candidate_counts( featuresMaps, commentLabels, flattenedFeaturesMap, CHist[-1] )
        logging.getLogger("CAR").info( str(len(CHist[-1])) + " Candidate " + str(idxK+1) + "-sequences have been generated.")
        FHist.append( [] )
        CAR_extract_frequent_rules( CHist[-1], minSup, minConf, n, FHist[-1] )
        logging.getLogger("CAR").info( str(len(FHist[-1])) + " Frequent " + str(idxK+1) + "-sequences have been generated.")
        
    #logging.getLogger("CAR").info( "CHist" + str(CHist) )
    #logging.getLogger("CAR").info( "FHist" + str(FHist) )
    
    # Serialize frequent lists to disk
    pickle.dump( FHist, open( cacheFileName, "wb" ) )
    
def CAR_conditional_apriori(ctx, featuresMaps, cacheFileName, minSup=0.1, minConf=0.5):
    logging.getLogger("CAR").info( "conditional apriori" )
    # See if cache exists
    if ( os.path.isfile(cacheFileName) == False ):
        CAR_apriori( ctx, featuresMaps, cacheFileName, minSup, minConf )
        
    FHist = pickle.load( open( cacheFileName ) )
    return FHist
    