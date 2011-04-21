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
import csv

def initLogger(): 
    logging.basicConfig( level=logging.DEBUG,
                         format='%(asctime)s %(levelname)s %(message)s',
                         filename='AmazonCommentMinerStatus.log',
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

def nFoldCrossValidation(n, csvCommentsPath, csvReviewsPath, classifierType):

    classifierPolicy = getClassifierPolicy( classifierType )
    
    rawCsvComments = csv.DictReader(open(csvCommentsPath))
    rawCsvComments = [comment for comment in rawCsvComments]
    random.shuffle(rawCsvComments);
    #rawCsvComments = rawCsvComments[0:100]
    
    rawCsvReviews = csv.DictReader(open(csvReviewsPath))
    rawCsvReviews = [review for review in rawCsvReviews]
    
    totalComments = len(rawCsvComments)
    nFoldness=n
    sectionSize=totalComments/nFoldness;
    for startIx in range(1,totalComments,sectionSize):
        # Pre-process data
        #ctxCacheFileName = "ctxCache.txt"
        trainSet = rawCsvComments[1:startIx]
        trainSet.extend(rawCsvComments[startIx+sectionSize:])
        
        trainInputs=getFeatureSet(None,trainSet,rawCsvReviews,classifierPolicy);
        testInputs=getFeatureSet(None,rawCsvComments[startIx+1:startIx+sectionSize-1],rawCsvReviews,classifierPolicy)
        
        # Test classifier
        classifier = classifierPolicy[ eClassifierCB.Classify ]( trainInputs, testInputs )

def getFeatureSet(ctxCacheFileName,rawCsvCommments,rawCsvReviews,classifierPolicy):
    ctx = MinerContext.loadContext(ctxCacheFileName, rawCsvCommments,rawCsvReviews,0.3 )
    return getFeatureSet2(ctx,classifierPolicy)

def getFeatureSet2(ctx,classifierPolicy):
    featuresMaps = []
    classifierPolicy[ eClassifierCB.PrepareFeatures ]( ctx, featuresMaps )
    return getFeatureSet3(ctx,featuresMaps,classifierPolicy)

def getFeatureSet3(ctx,featuresMaps,classifierPolicy):
    classifierInputs = []
    classifierPolicy[ eClassifierCB.ClassifierInputs ]( ctx, featuresMaps, classifierInputs )
    return classifierInputs
    
def writeOutput(ctxTest,featuresMapsTest,classifierType,classifier):
    errorFile=open("errors.txt",'w')
    outputFile=open("output.txt",'w')
    if(classifierType == eClassifierType.NaiveBayes):
        testRawCsvComments = ctxTest.mRawCsvComments #csv.DictReader(open("../data/testing-data.csv"))
        #testRawCsvComments = [comment for comment in testRawCsvComments]
        count = 1
        logging.getLogger("NaiveBayes").info( "show mis-classified" )
        for itrComment, rawCsvCommentDict in enumerate( testRawCsvComments ):
            probDist = classifier.prob_classify(featuresMapsTest[itrComment])
            bClassifierIsPositive = probDist.prob('1')>0.5
            bClassifierIsNegative = probDist.prob('0')>0.5
            bCommentIsNegative = (rawCsvCommentDict[ "Thumbs Up!" ]=='0' and rawCsvCommentDict[ "Thumbs Down" ]=='0')
            bCommentIsPositive = (rawCsvCommentDict[ "Thumbs Up!" ]=='1' or rawCsvCommentDict[ "Thumbs Down" ]=='1') 
            if( ( bClassifierIsPositive and bCommentIsNegative ) or ( bClassifierIsNegative and bCommentIsPositive ) ):
                errorFile.write(str("#" + str(count) + " \r\n 0:" + str(probDist.prob('0')) + " 1:" + str(probDist.prob('1'))) + "\r\n")
                errorFile.write(str(rawCsvCommentDict["Comment"]) + "\r\n\r\n")
                #errorFile.write(str(featuresMaps[itrComment]) + "\r\n\r\n")
                count+=1
            
            if(bClassifierIsPositive) :
                outputFile.write("<" + str(rawCsvCommentDict["Comment_ID"]) + "><1>\r\n")
            else:
                outputFile.write("<" + str(rawCsvCommentDict["Comment_ID"]) + "><0>\r\n")

def trainAndClassify(csvCommentsPath,csvCommentsPathTest,csvReviewsPath,classifierType):
    classifierPolicy = getClassifierPolicy( classifierType )
    
    rawCsvComments = csv.DictReader(open(csvCommentsPath))
    rawCsvComments = [comment for comment in rawCsvComments]
    #rawCsvComments = rawCsvComments[0:10]
    
    rawCsvCommentsTest = csv.DictReader(open(csvCommentsPathTest))
    rawCsvCommentsTest = [comment for comment in rawCsvCommentsTest]
    #rawCsvCommentsTest = rawCsvCommentsTest[0:10]
    
    rawCsvReviews = csv.DictReader(open(csvReviewsPath))
    rawCsvReviews = [review for review in rawCsvReviews]
    
    trainInputs=getFeatureSet(None,rawCsvComments,rawCsvReviews,classifierPolicy);
    
    ctxTest = MinerContext.loadContext(None, rawCsvCommentsTest,rawCsvReviews,0.3 )
    featuresMapsTest = []
    classifierPolicy[ eClassifierCB.PrepareFeatures ]( ctxTest, featuresMapsTest )
    
    testInputs=getFeatureSet3(ctxTest,featuresMapsTest,classifierPolicy)
    
    classifier = classifierPolicy[ eClassifierCB.Classify ]( trainInputs, testInputs )

    writeOutput(ctxTest,featuresMapsTest,classifierType,classifier)

def appMain():
    
    # Select our classifier policy
    classifierType = eClassifierType.NaiveBayes
        
    # initialize our logger
    initLogger()

    csvCommentsPath="../data/training-data.csv"
    csvCommentsPathTest="../data/testing-data.csv"
    csvReviewsPath="../data/all-reviews.csv"
    
    nFoldCrossValidation(5, csvCommentsPath, csvReviewsPath, classifierType)
    #trainAndClassify(csvCommentsPath,csvCommentsPathTest,csvReviewsPath,classifierType)
    

if __name__ == '__main__':
    appMain()