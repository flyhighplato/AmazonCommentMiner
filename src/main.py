'''
main.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerContext
import MinerNaiveBayes
import MinerSvm
import random

def initLogger(): 
    logging.basicConfig( level=logging.DEBUG,
                         format='%(asctime)s %(levelname)s %(message)s',
                         filename='msgsp.log',
                         filemode='w' )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# Function callback offsets 
class eClassifierCB:
    PrepareFeatures  = 0
    ClassifierInputs = 1
    Classify         = 2

# Enumerated classifier types
class eClassifierType:
    NaiveBayes = 0
    Svm = 1
    
def getClassifierPolicy( classifierType ):
    ClassifierPolices = [ MinerNaiveBayes.NaiveBayesGetPolicy(), MinerSvm.SvmGetPolicy() ]
    return ClassifierPolices[ classifierType ]

def appMain():
    
    # Select our classifier policy
    classifierType = eClassifierType.NaiveBayes
    classifierPolicy = getClassifierPolicy( classifierType )
    
    # initialize our logger
    initLogger()
    
    # Pre-process data
    ctxCacheFileName = "ctxCache.txt"
    ctx = MinerContext.loadContext(ctxCacheFileName, "../data/training-data.csv","../data/all-reviews.csv", 10, 1100, 0.3 )
    
    # Map comments to features sets
    featuresMaps = []
    classifierPolicy[ eClassifierCB.PrepareFeatures ]( ctx, featuresMaps )
    
    # Convert to classifier input
    classifierInputs = []
    classifierPolicy[ eClassifierCB.ClassifierInputs ]( ctx, featuresMaps, classifierInputs )

    # Shuffle classifier inputs
    random.shuffle( classifierInputs )
    trainInputs, testInputs = classifierInputs[ len( classifierInputs )/2: ], classifierInputs[ :len(classifierInputs)/2 ]

    errorFile=open("errors.txt",'w')
    # Test classifier
    classifier = classifierPolicy[ eClassifierCB.Classify ]( trainInputs, testInputs )
    
    count = 1
    logging.getLogger("NaiveBayes").info( "show mis-classified" )
    for itrComment, rawCsvCommentDict in enumerate( ctx.mRawCsvComments ):
        probDist = classifier.prob_classify(featuresMaps[itrComment])
        bClassifierIsPositive = probDist.prob('1')>0.5
        bClassifierIsNegative = probDist.prob('0')>0.5
        bCommentIsNegative = (rawCsvCommentDict[ "Thumbs Up!" ]=='0' and rawCsvCommentDict[ "Thumbs Down" ]=='0')
        bCommentIsPositive = (rawCsvCommentDict[ "Thumbs Up!" ]=='1' or rawCsvCommentDict[ "Thumbs Down" ]=='1') 
        if( ( bClassifierIsPositive and bCommentIsNegative ) or ( bClassifierIsNegative and bCommentIsPositive ) ):
            errorFile.write(str("#" + str(count) + " \r\n 0:" + str(probDist.prob('0')) + " 1:" + str(probDist.prob('1'))) + "\r\n")
            errorFile.write(str(rawCsvCommentDict["Comment"]) + "\r\n\r\n")
            #errorFile.write(str(featuresMaps[itrComment]) + "\r\n\r\n")
            count+=1

if __name__ == '__main__':
    appMain()