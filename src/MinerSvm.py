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
import MinerMiscUtils
import os

import subprocess

def SvmPrepareFeatures( ctx, outFeaturesMaps ):
    logging.getLogger("Svm").info( "prepare features" )
    MinerFeaturesUtils.initFeatures( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesCommentLength( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesHelpfulnessRatio( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesPhrases( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesWordExists( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesAuthorFreqInReview(ctx, outFeaturesMaps)
    MinerFeaturesUtils.addFeaturesReviewAuthorMentioned(ctx, outFeaturesMaps)
    MinerFeaturesUtils.addFeaturesCommentAuthorMentioned( ctx, outFeaturesMaps )
    MinerFeaturesUtils.addFeaturesCAR( ctx, outFeaturesMaps )

def SvmUtilGetStrSign( value ):
    if ( value >= 0 ):
        return "+"
    return ""
                
def SvmGetClassifierInputs( ctx, featuresMaps, outClassifierInputs ):
    logging.getLogger("Svm").info( "get classifier inputs" )
    outClassifierInputs[:] = []
    
    featuresKeys = set()
    
    for featureVector in featuresMaps:
        featuresKeys.update(featureVector.keys())
    #featuresKeys = featuresMaps[0].keys() # Assuming at least a single features map exists
    for itrComment, rawCsvCommentDict in enumerate( ctx.mRawCsvComments ):
        # @TODO: Classify "Thumbs Down!"
        svmType = -1 + 2 * MinerMiscUtils.getCommentLabel( rawCsvCommentDict )
        inputsCollector = [SvmUtilGetStrSign( svmType ) + str(svmType)]
        for itrFeature, featureKey in enumerate( featuresKeys ):
            if(featureKey in featuresMaps[itrComment]):
                featureValue = -1 + 2*int(featuresMaps[itrComment][ featureKey ])
            else:
                featureValue=0
                
            inputsCollector.append( " " + str( itrFeature+1 ) + ":" + str(featureValue) )
            
        outClassifierInputs.append( "".join( inputsCollector ) )
    assert( len( outClassifierInputs ) == len( ctx.mRawCsvComments ) )    

def SvmClassify( trainInputs, testInputs ):
    logging.getLogger("Svm").info( "classifying" )
    
    svmOutput = open("svmTrain.txt", "w")
    for input in trainInputs:
        svmOutput.write(input + "\r\n")
        
    svmOutput = open("svmTest.txt", "w")
    for input in testInputs:
        svmOutput.write(input + "\r\n")
    
    #This variable is required on OS X 64-bit.  Probably isnt' a problem anywhere else?
    os.putenv('VERSIONER_PYTHON_PREFER_32_BIT','yes')
    
    #Runs libsvm and redirects its output
    easyPy = subprocess.Popen(['python', "../libs/libsvm-3.0/tools/easy.py",'svmTrain.txt','svmTest.txt'], 
                        stdout=subprocess.PIPE,
                        )
    for line in easyPy.stdout:
        print line.strip()
    
    return 0

def SvmGetPolicy():
    return [ SvmPrepareFeatures, SvmGetClassifierInputs, SvmClassify ]
