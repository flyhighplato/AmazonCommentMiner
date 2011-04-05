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
    ctx = MinerContext.Context( "../data/training-data.csv", 10, 1100, 0.3 )
    
    # Map comments to features sets
    featuresMaps = []
    classifierPolicy[ eClassifierCB.PrepareFeatures ]( ctx, featuresMaps )
    
    # Convert to classifier input
    classifierInputs = []
    classifierPolicy[ eClassifierCB.ClassifierInputs ]( ctx, featuresMaps, classifierInputs )

    # Shuffle classifier inputs
    random.shuffle( classifierInputs )
    trainInputs, testInputs = classifierInputs[ len( classifierInputs )/2: ], classifierInputs[ :len(classifierInputs)/2 ]

    # Test classifier
    classifierPolicy[ eClassifierCB.Classify ]( trainInputs, testInputs )

if __name__ == '__main__':
    appMain()