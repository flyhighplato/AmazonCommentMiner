'''
MinerSvm.py

@author: garyturovsky
@author: alanperezrathke
'''

'''
MinerNaiveBayes.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerFeaturesUtils

def SvmPrepareFeatures( ctx, outFeaturesMaps ):
    logging.getLogger("Svm").info( "prepare features" )
    MinerFeaturesUtils.initFeatures( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesCommentLength( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesHelpfulnessRatio( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesWordExists( ctx, outFeaturesMaps )

def SvmUtilGetStrSign( value ):
    if ( value >= 0 ):
        return "+"
    return ""
                
def SvmGetClassifierInputs( ctx, featuresMaps, outClassifierInputs ):
    logging.getLogger("Svm").info( "get classifier inputs" )
    outClassifierInputs[:] = []
    featuresKeys = featuresMaps[0].keys() # Assuming at least a single features map exists
    for itrComment, rawCsvCommentDict in enumerate( ctx.mRawCsvComments ):
        # @TODO: Classify "Thumbs Down!"
        svmType = -1 + 2 * int( rawCsvCommentDict[ "Thumbs Up!" ] )
        inputsCollector = [SvmUtilGetStrSign( svmType ) + str(svmType)]
        for itrFeature, featureKey in enumerate( featuresKeys ):
            featureValue = featuresMaps[itrComment][ featureKey ]
            inputsCollector.append( " " + str( itrFeature+1 ) + ":" + str(featureValue) )
        outClassifierInputs.append( "".join( inputsCollector ) )
    assert( len( outClassifierInputs ) == len( ctx.mRawCsvComments ) )    
        
def SvmClassify( trainInputs, testInputs ):
    # @ TODO:
    return 0