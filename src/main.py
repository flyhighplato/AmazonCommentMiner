'''
main.py

@author: garyturovsky
@author: alanperezrathke
'''

import logging
import MinerContext
import MinerNaiveBayes
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

def appMain():
    
    # initialize our logger
    initLogger()
    
    # Pre-process data
    ctx = MinerContext.Context( "../data/training-data.csv", 10, 1100, 0.3 )
    
    # Map comments to features sets
    featuresMaps = []
    MinerNaiveBayes.NaiveBayesPrepareFeatures( ctx, featuresMaps )
    
    # Convert to classifier input
    classifierInputs = []
    MinerNaiveBayes.NaiveBayesGetClassifierInputs( ctx, featuresMaps, classifierInputs )

    # Shuffle classifier inputs
    random.shuffle( classifierInputs )
    trainInputs, testInputs = classifierInputs[ len( classifierInputs )/2: ], classifierInputs[ :len(classifierInputs)/2 ]

    # Test classifier
    MinerNaiveBayes.NaiveBayesClassify( trainInputs, testInputs )

if __name__ == '__main__':
    appMain()