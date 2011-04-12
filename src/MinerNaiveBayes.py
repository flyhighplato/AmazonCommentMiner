'''
MinerNaiveBayes.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerFeaturesUtils
import nltk
    
def NaiveBayesPrepareFeatures( ctx, outFeaturesMaps ):
    logging.getLogger("NaiveBayes").info( "prepare features" )
    MinerFeaturesUtils.initFeatures( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesCommentLength( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesHelpfulnessRatio( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesPhrases( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesWordExists( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesAuthorFreqInReview(ctx, outFeaturesMaps)
    
def NaiveBayesGetClassifierInputs( ctx, featuresMaps, outClassifierInputs ):
    logging.getLogger("NaiveBayes").info( "get classifier inputs" )
    outClassifierInputs[:] = []
    for itrComment, rawCsvCommentDict in enumerate( ctx.mRawCsvComments ):
        # @TODO: Classify "Thumbs Down!"
        type = str(int(rawCsvCommentDict[ "Thumbs Up!" ]) or int(rawCsvCommentDict[ "Thumbs Down" ]))
        outClassifierInputs.append( ( featuresMaps[ itrComment ], type ) )
        
def NaiveBayesClassify( trainInputs, testInputs ):
    logging.getLogger("NaiveBayes").info( "classify" )
    classifier = nltk.NaiveBayesClassifier.train( trainInputs )
    print nltk.classify.accuracy( classifier, testInputs )
    classifier.show_most_informative_features( 1000 );
    
    return classifier

def NaiveBayesGetPolicy():
    return [ NaiveBayesPrepareFeatures, NaiveBayesGetClassifierInputs, NaiveBayesClassify ]
