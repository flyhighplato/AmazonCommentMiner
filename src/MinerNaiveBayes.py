'''
MinerNaiveBayes.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerMiscUtils
import MinerFeaturesUtils
import nltk

def NaiveBayesPrepareFeatures( ctx, outFeaturesMaps, featuresBitMask ):
    logging.getLogger("NaiveBayes").info( "prepare features" )
    MinerFeaturesUtils.initFeatures( ctx, outFeaturesMaps )
    if ( featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.wordExists ):
        MinerFeaturesUtils.addFeaturesWordExists( ctx, outFeaturesMaps )
    if ( featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.commentLength ):
        MinerFeaturesUtils.addFeaturesCommentLength( ctx, outFeaturesMaps )
    if ( featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.helpfullnessRatio ):
        MinerFeaturesUtils.addFeaturesHelpfulnessRatio( ctx, outFeaturesMaps )
    if ( featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.authorFreqInReview ):
        MinerFeaturesUtils.addFeaturesAuthorFreqInReview(ctx, outFeaturesMaps)
    if ( featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.reviewAuthorMentioned ):
        MinerFeaturesUtils.addFeaturesReviewAuthorMentioned(ctx, outFeaturesMaps)
    if ( featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.commentAuthorMentioned ):
        MinerFeaturesUtils.addFeaturesCommentAuthorMentioned( ctx, outFeaturesMaps )
    if ( featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.dist ):
        MinerFeaturesUtils.addFeaturesDist( ctx, outFeaturesMaps )
    if ( featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.phrases ):
        MinerFeaturesUtils.addFeaturesPhrases( ctx, outFeaturesMaps )
    
def NaiveBayesGetClassifierInputs( ctx, featuresMaps, outClassifierInputs, bTrain ):
    logging.getLogger("NaiveBayes").info( "get classifier inputs" )
    outClassifierInputs[:] = []
    for itrComment, rawCsvCommentDict in enumerate( ctx.mRawCsvComments ):
        strLabel = "?"
        if ( bTrain ):
            strLabel = str(MinerMiscUtils.getCommentLabel(rawCsvCommentDict))
        outClassifierInputs.append( ( featuresMaps[ itrComment ], strLabel ) )

def NaiveBayesClassify( trainInputs, testInputs, bDebug, outDebugFileName, outDebugLabel ):
    logging.getLogger("NaiveBayes").info( "classify" )
    classifier = nltk.NaiveBayesClassifier.train( trainInputs )
    if ( bDebug ):
        accuracy = nltk.classify.accuracy( classifier, testInputs )
        fileHandle = open ( outDebugFileName, 'a' )
        fileHandle.write ( outDebugLabel + "," + str(accuracy) + '\n' )
        fileHandle.close()
        print "Accuracy = " + str(accuracy) + '\n'
        classifier.show_most_informative_features( 10 );
    return classifier

def NaiveBayesGetPolicy():
    return [ NaiveBayesPrepareFeatures, NaiveBayesGetClassifierInputs, NaiveBayesClassify ]
