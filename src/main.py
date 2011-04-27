'''
main.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerContext
import MinerNaiveBayes
import MinerFeaturesUtils
import MinerSvm
import random
import csv
import copy

def initLogger(): 
    logging.basicConfig(level=logging.DEBUG,
                         format='%(asctime)s %(levelname)s %(message)s',
                         filename='AmazonCommentMinerStatus.log',
                         filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# Function callback offsets 
class eClassifierCB:
    PrepareFeatures = 0
    ClassifierInputs = 1
    Classify = 2

# Enumerated classifier types
class eClassifierType:
    NaiveBayes = 0
    Svm = 1
    
def getClassifierPolicy(classifierType):
    ClassifierPolices = [ MinerNaiveBayes.NaiveBayesGetPolicy(), MinerSvm.SvmGetPolicy() ]
    return ClassifierPolices[ classifierType ]
    
def writeOutput(ctxTest, featuresMapsTest, classifierType, classifier):
    #errorFile = open("errors.txt", 'w')
    outputFile = open("output.txt", 'w')
    if(classifierType == eClassifierType.NaiveBayes):
        testRawCsvComments = ctxTest.mRawCsvComments #csv.DictReader(open("../data/testing-data.csv"))
        #testRawCsvComments = [comment for comment in testRawCsvComments]
        #count = 1
        logging.getLogger("NaiveBayes").info("show mis-classified")
        for itrComment, rawCsvCommentDict in enumerate(testRawCsvComments):
            probDist = classifier.prob_classify(featuresMapsTest[itrComment])
            bClassifierIsPositive = probDist.prob('1') > 0.5
            #bClassifierIsNegative = probDist.prob('0') > 0.5
            #bCommentIsNegative = (rawCsvCommentDict[ "Thumbs Up!" ] == '0' and rawCsvCommentDict[ "Thumbs Down" ] == '0')
            #bCommentIsPositive = (rawCsvCommentDict[ "Thumbs Up!" ] == '1' or rawCsvCommentDict[ "Thumbs Down" ] == '1') 
            #if((bClassifierIsPositive and bCommentIsNegative) or (bClassifierIsNegative and bCommentIsPositive)):
            #    errorFile.write(str("#" + str(count) + " \r\n 0:" + str(probDist.prob('0')) + " 1:" + str(probDist.prob('1'))) + "\r\n")
            #    errorFile.write(str(rawCsvCommentDict["Comment"]) + "\r\n\r\n")
                #errorFile.write(str(featuresMaps[itrComment]) + "\r\n\r\n")
            #    count += 1
            if(bClassifierIsPositive) :
                assert(probDist.prob('0') < probDist.prob('1'))
                outputFile.write("(" + str(rawCsvCommentDict["Comment_ID"]) + " 1)\r\n")
            else:
                assert(probDist.prob('0') > probDist.prob('1'))
                outputFile.write("(" + str(rawCsvCommentDict["Comment_ID"]) + " 0)\r\n")

# Structure for easier passing of train and classify parameters
class TrainAndClassifyParams:
    
    # Constructor
    def __init__( self, csvCommentsPathTrain="", csvCommentsPathTest="", csvReviewsPath="", classifierType=0, featuresBitMask=0, ctxCacheTrainFileName="", ctxCacheTestFileName="", supportThresh=0, bDebug=0, outDebugFileName="", outDebugLabel="", CARMinSup=0, CARMinConf=0, CARCacheFileName="" ):
        self.csvCommentsPathTrain = csvCommentsPathTrain
        self.csvCommentsPathTest = csvCommentsPathTest
        self.csvReviewsPath = csvReviewsPath
        self.classifierType = classifierType
        self.featuresBitMask = featuresBitMask
        self.ctxCacheTrainFileName = ctxCacheTrainFileName
        self.ctxCacheTestFileName = ctxCacheTestFileName
        self.supportThresh = supportThresh
        self.bDebug = bDebug
        self.outDebugFileName = outDebugFileName
        self.outDebugLabel = outDebugLabel
        self.CARMinSup = CARMinSup
        self.CARMinConf = CARMinConf
        self.CARCacheFileName = CARCacheFileName
        

def trainAndClassify( params ):
    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~BEGIN" )
    classifierPolicy = getClassifierPolicy(params.classifierType)
    
    # Load raw CSV reviews
    rawCsvReviews = csv.DictReader(open(params.csvReviewsPath))
    rawCsvReviews = [review for review in rawCsvReviews]

    # TRAIN: Load raw CSV comments
    rawCsvCommentsTrain = csv.DictReader(open(params.csvCommentsPathTrain))
    rawCsvCommentsTrain = [comment for comment in rawCsvCommentsTrain]
                
    # TRAIN: Create context
    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~CREATING TRAINING CONTEXT" )
    ctxTrain = MinerContext.loadContext(params.ctxCacheTrainFileName, rawCsvCommentsTrain, rawCsvReviews, params.supportThresh)

    # TRAIN: Create features sets    
    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~PREPARING TRAINING FEATURES" )
    featuresMapsTrain = []
    classifierPolicy[ eClassifierCB.PrepareFeatures ](ctxTrain, featuresMapsTrain, params.featuresBitMask)
    
    # TRAIN: Add CAR if desired
    if ( params.featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.CAR ):
        MinerFeaturesUtils.addFeaturesCAR( ctxTrain, featuresMapsTrain, params.CARMinSup, params.CARMinConf, params.CARCacheFileName )

    # TRAIN: Convert features set to classifier specific input
    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~CONVERTING TRAINING INPUTS" )
    classifierInputsTrain = []
    classifierPolicy[ eClassifierCB.ClassifierInputs ](ctxTrain, featuresMapsTrain, classifierInputsTrain, True)

    # TEST: Load raw CSV comments
    rawCsvCommentsTest = csv.DictReader(open(params.csvCommentsPathTest))
    rawCsvCommentsTest = [comment for comment in rawCsvCommentsTest]

    # TEST: Create context
    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~CREATING TESTING CONTEXT" )
    
    ctxTest = MinerContext.loadContext(params.ctxCacheTestFileName, rawCsvCommentsTest, rawCsvReviews, params.supportThresh)
    # HACK - replace filtered words with those of training context
    ctxTest.mFilteredWords = ctxTrain.mFilteredWords

    # TEST: Create features sets    
    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~PREPARING TESTING FEATURES" )
    featuresMapsTest = []
    classifierPolicy[ eClassifierCB.PrepareFeatures ](ctxTest, featuresMapsTest, params.featuresBitMask )
    
    # TEST: Add CAR if desired
    if ( params.featuresBitMask & MinerFeaturesUtils.eFeaturesMaskBits.CAR ):
        MinerFeaturesUtils.addFeaturesCAR( ctxTest, featuresMapsTest, params.CARMinSup, params.CARMinConf, params.CARCacheFileName )
    
    # TEST: Convert features set to classifier specific input
    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~CONVERTING TESTING INPUTS" )
    classifierInputsTest = []
    classifierPolicy[ eClassifierCB.ClassifierInputs ](ctxTest, featuresMapsTest, classifierInputsTest, True)

    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~CLASSIFYING" )
    classifier = classifierPolicy[ eClassifierCB.Classify ]( classifierInputsTrain, classifierInputsTest, params.bDebug, params.outDebugFileName, params.outDebugLabel )

    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~WRITING OUTPUTS" )
    writeOutput(ctxTest, featuresMapsTest, params.classifierType, classifier)
    logging.getLogger( "trainAndClassify" ).info( "~~~~~~~~~~~~~END" )

def standardRun():
    # initialize our logger
    initLogger()
    logging.getLogger( "Standard Run" ).info( "BEGIN" )    
    params = TrainAndClassifyParams()
    
    # Select our classifier policy
    params.classifierType = eClassifierType.NaiveBayes
    
    # Determine train and test paths
    params.csvCommentsPathTrain = "../data/training-data.csv"
    params.csvCommentsPathTest = "../data/testing-data.csv"
    params.csvReviewsPath = "../data/all-reviews.csv"
    
    params.ctxCacheTrainFileName = "ctxCacheTrain.txt"
    params.ctxCacheTestFileName = "ctxCacheTest.txt"
    params.supportThresh = 0.3
    params.bDebug = True
    params.outDebugFileName = "perf.csv"
    params.outDebugLabel = "Standard Run"
    
    params.CARCacheFileName = "CARcache.txt"
    params.CARMinSup = 0.1
    params.CARMinConf = 0.6
    
    params.featuresBitMask = MinerFeaturesUtils.eFeaturesMaskBits.wordExists | MinerFeaturesUtils.eFeaturesMaskBits.authorFreqInReview | MinerFeaturesUtils.eFeaturesMaskBits.phrases | MinerFeaturesUtils.eFeaturesMaskBits.CAR
    
    trainAndClassify( params )
    logging.getLogger( "Standard Run" ).info( "END" )

def wordCountSupportExperiment():
    # initialize our logger
    initLogger()

    logging.getLogger( "Word Count Support Experiment" ).info( "BEGIN" )
    params = TrainAndClassifyParams()
    
    # Select our classifier policy
    params.classifierType = eClassifierType.NaiveBayes
        
    # Determine train and test paths
    params.csvCommentsPathTrain = "../data/training-data.csv"
    params.csvCommentsPathTest = "../data/testing-data.csv"
    params.csvReviewsPath = "../data/all-reviews.csv"
    
    params.ctxCacheTrainFileName = None
    params.ctxCacheTestFileName = "ctxCacheTest.txt"
    params.bDebug = True
    
    params.featuresBitMask = MinerFeaturesUtils.eFeaturesMaskBits.wordExists | MinerFeaturesUtils.eFeaturesMaskBits.phrases
    params.outDebugFileName = "../data/wordSupportCountsExperiment.csv"
    
    params.CARCacheFileName = "CARcache.txt"
    params.CARMinSup = 0.1
    params.CARMinConf = 0.6
        
    params.supportThresh = 0.0
    while ( params.supportThresh <= 1.0 ):
        params.outDebugLabel = str(params.supportThresh)
        trainAndClassify( params )
        params.supportThresh += 0.1
        
    logging.getLogger( "Word Count Support Experiment" ).info( "END" )

def CARSupExperiment():
    # initialize our logger
    initLogger()
    logging.getLogger( "CAR Support Experiment" ).info( "BEGIN" )
    params = TrainAndClassifyParams()
    
    # Select our classifier policy
    params.classifierType = eClassifierType.NaiveBayes
        
    # Determine train and test paths
    params.csvCommentsPathTrain = "../data/training-data.csv"
    params.csvCommentsPathTest = "../data/testing-data.csv"
    params.csvReviewsPath = "../data/all-reviews.csv"
    
    params.ctxCacheTrainFileName = "ctxCacheTrain.txt"
    params.ctxCacheTestFileName = "ctxCacheTest.txt"
    params.bDebug = True
    
    params.featuresBitMask = MinerFeaturesUtils.eFeaturesMaskBits.wordExists | MinerFeaturesUtils.eFeaturesMaskBits.authorFreqInReview | MinerFeaturesUtils.eFeaturesMaskBits.phrases | MinerFeaturesUtils.eFeaturesMaskBits.CAR
    params.supportThresh = 0.3
    
    # Run CAR support experiment 
    params.outDebugFileName = "../data/CARSupExperiment.csv"
    params.CARMinConf = 0.6
    
    params.CARMinSup = 0.025
    while ( params.CARMinSup <= 0.5 ):
        params.outDebugLabel = str(params.CARMinSup)
        params.CARCacheFileName = "CARSupCache"+params.outDebugLabel+".txt"
        trainAndClassify( params )
        params.CARMinSup += 0.025
        
    logging.getLogger( "CAR Support Experiment" ).info( "END" )

def CARConfExperiment():
    # initialize our logger
    initLogger()
    logging.getLogger( "CAR Confidence Experiment" ).info( "BEGIN" )
    params = TrainAndClassifyParams()
    
    # Select our classifier policy
    params.classifierType = eClassifierType.NaiveBayes
        
    # Determine train and test paths
    params.csvCommentsPathTrain = "../data/training-data.csv"
    params.csvCommentsPathTest = "../data/testing-data.csv"
    params.csvReviewsPath = "../data/all-reviews.csv"
    
    params.ctxCacheTrainFileName = "ctxCacheTrain.txt"
    params.ctxCacheTestFileName = "ctxCacheTest.txt"
    params.bDebug = True
    
    params.featuresBitMask = MinerFeaturesUtils.eFeaturesMaskBits.wordExists | MinerFeaturesUtils.eFeaturesMaskBits.authorFreqInReview | MinerFeaturesUtils.eFeaturesMaskBits.phrases | MinerFeaturesUtils.eFeaturesMaskBits.CAR
    params.supportThresh = 0.3
    
    # Run CAR support experiment 
    params.outDebugFileName = "../data/CARConfExperiment.csv"
    params.CARMinSup = 0.1
    
    params.CARMinConf = 0.55
    while ( params.CARMinSup <= 0.75 ):
        params.outDebugLabel = str(params.CARMinConf)
        params.CARCacheFileName = "CARConfCache"+params.outDebugLabel+".txt"
        trainAndClassify( params )
        params.CARMinConf += 0.01

    logging.getLogger( "CAR Confidence Experiment" ).info( "END" )
    
def appMain():
    #standardRun()
    #wordCountSupportExperiment()
    #CARSupExperiment()
    CARConfExperiment()
    
if __name__ == '__main__':
    appMain()
