import math
import sys
import io

class NaiveBayes(object):

    def countDocProbsAndClassify(self,speechLst):
        redCount = 0
        blueCount = 0
        d = {}
        d["RED"] = []
        d["BLUE"] = []
        #print(f'speechLst {speechLst} \n',speechLst)
        for speech in speechLst:
            splited = speech.split("\t")
            c = splited[0]
            #print(f'speech {speech} \n',speech)
            theSpeech = splited[1]
            if c == "RED":
                d["RED"].append(theSpeech)
                redCount += 1
            else:
                d["BLUE"].append(theSpeech)
                blueCount += 1
        redProb = redCount / (redCount + blueCount)
        blueProb = 1 - redProb
        return (math.log(redProb),math.log(blueProb),d)

    def __init__(self, trainingData):
        """
        Initializes a NaiveBayes object and the naive bayes classifier.
        :param trainingData: full text from training file as a single large string
        """
        '''
        wordLst = trainingData.split(" ")
        with open "train.txt" as f:
            testData = f.read()
        '''
        speechLst = trainingData.split("\n")
        redProb,blueProb,d = self.countDocProbsAndClassify(speechLst)
        blueSpeeches = "".join(d["BLUE"])
        redSpeeches = "".join(d["RED"])
        blueWordLst = blueSpeeches.split(" ")
        redWordLst = redSpeeches.split(" ")
        blueLen = len(blueWordLst)
        redLen = len(redWordLst)
        blueUniqueWords = set(blueWordLst)
        redUniqueWords = set(redWordLst)
        vocabSize = len(set(redUniqueWords.union(blueUniqueWords)))
        blueD = {}
        redD = {}

        for word in blueUniqueWords:
            blueD[word] = (blueWordLst.count(word)+1) / (blueLen+vocabSize)

        for word in redUniqueWords:
            redD[word] = (redWordLst.count(word)+1) / (redLen+vocabSize)

        self.blueProb = blueProb
        self.redProb = redProb
        self.blueD = blueD
        self.redD = redD
        self.blueLen = blueLen
        self.redLen = redLen
        self.vocabSize = vocabSize

    def estimateLogProbability(self, sentence):
        """
        Using the naive bayes model generated in __init__, calculate the probabilities that this sentence is in each category. Sentence is a single sentence from the test set. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary. 
        :param sentence: the test sentence, as a single string without label
        :return: a dictionary containing log probability for each category
        """

        wordLst = sentence.split(" ")
        redProb = 0
        blueProb = 0
        for word in wordLst:
            if word in self.redD:
                redProb += math.log(self.redD[word])
            else:
                redProb += math.log(1/(self.redLen+self.vocabSize))
            if word in self.blueD:
                blueProb += math.log(self.blueD[word])
            else:
                blueProb += math.log(1/(self.blueLen+self.vocabSize))
        return {'red': redProb+self.redProb, 'blue': blueProb+self.blueProb}

    def testModel(self, testData):
        """
        Using the naive bayes model generated in __init__, test the model using the test data. You should calculate accuracy, precision for each category, and recall for each category. 
        This function is required by the autograder. Please do not delete or modify the name of this function. Please do not change the name of each key in the dictionary.
        :param testData: the test file as a single string
        :return: a dictionary containing each item as identified by the key
        """
        '''
        with open("train.txt") as f:
            trainingData = f.read()
        obj = NaiveBayes(trainingData)
        '''
#        print(f'test data {testData}',testData)
        markedRed = 0
        markedBlue = 0
        hit = 0
        miss = 0
        blueHit = 0
        blueMiss = 0
        redHit = 0
        redMiss = 0
        sentenceLst = testData.split("\n")
        for speech in sentenceLst:
            tmp = speech.split("\t")
            label = tmp[0]
            speech = tmp[1]
            d = self.estimateLogProbability(speech)
            redProb = d["red"]
            blueProb = d["blue"]
            diff = abs(redProb - blueProb)
            #print("confidence {diff}\n",diff)
            right = False
            if redProb > blueProb:
                markedRed += 1
                if label == "RED":
                    right = True
                    hit += 1
                    redHit += 1 # True Positive
                else:
                    miss += 1
                    redMiss += 1 # False Positive & blue False Negative
            else:
                markedBlue += 1
                if label == "BLUE":
                    right = True
                    hit += 1
                    blueHit += 1 # True Positive
                else:
                    miss += 1
                    blueMiss += 1 # blue False Positive & red False Negative
            #print(right)
        overallAccuracy = hit / (hit+miss)
        redPrecision = redHit / (redHit + redMiss)
        bluePrecision = blueHit / (blueHit + blueMiss)
        redRecall = redHit / (redHit + blueMiss)
        blueRecall = blueHit / (blueHit + redMiss)

        # model solution has an accuracy of > 0.90 for the first dataset and > 0.75 for the second dataset

        return {'overall accuracy': overallAccuracy,
                'precision for red': redPrecision,
                'precision for blue': bluePrecision,
                'recall for red': redRecall,
                'recall for blue': blueRecall}

"""
The following code is used only on your local machine. The autograder will only use the functions in the NaiveBayes class.            

You are allowed to modify the code below. But your modifications will not be used on the autograder.
"""
if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage: python3 naivebayes.py TRAIN_FILE_NAME TEST_FILE_NAME")
        sys.exit(1)

    train_txt = sys.argv[1]
    test_txt = sys.argv[2]

    with io.open(train_txt, 'r', encoding='utf8') as f:
        train_data = f.read()

    with io.open(test_txt, 'r', encoding='utf8') as f:
        test_data = f.read()

    model = NaiveBayes(train_data)
    evaluation = model.testModel(test_data)
    print("overall accuracy: " + str(evaluation['overall accuracy'])
        + "\nprecision for red: " + str(evaluation['precision for red'])
        + "\nprecision for blue: " + str(evaluation['precision for blue'])
        + "\nrecall for red: " + str(evaluation['recall for red'])
        + "\nrecall for blue: " + str(evaluation['recall for blue']))



