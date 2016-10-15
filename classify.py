from abc import ABCMeta, abstractmethod
import time

class Classifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, inputFile):
        return NotImplemented

    @abstractmethod
    def classify(self, inputVector, *args):
        return NotImplemented

    def tokenize(self, text):
        return [i.lower().strip() for i in text.split()]

    def parse(self, inputLine):
        rawFeatures = inputLine.strip().split("\t")
        return (rawFeatures[2].strip(), self.tokenize(rawFeatures[0])), rawFeatures[1].strip()

    def parseTesting(self, inputLine):
        rawFeatures = inputLine.strip().split("\t")
        return (rawFeatures[2], tokenize(rawFeaures[0]))

class NB(Classifier):
    def __init__(self):
        self.tokenFreqGivenLabel = {}
        self.vocabGivenLabel = {}
        self.labelFreq = {}
        self.trainingSize = 0.0
        self.totalTokens = 0.0
        self.categoryFilter = {}
        self.likelihoods = {}
        self.priors = {}

    def update(self, featureVector, label):
        category, tokens = featureVector
        if category not in self.tokenFreqGivenLabel:
            self.tokenFreqGivenLabel[category] = {}
            self.vocabGivenLabel[category] = {}
            self.labelFreq[category] = {}
            self.likelihoods[category] = {}
            self.priors[category] = {}
        if label not in self.tokenFreqGivenLabel[category]:
            self.tokenFreqGivenLabel[category][label] = {}
            self.vocabGivenLabel[category][label] = 0
            self.labelFreq[category][label] = 0
            self.likelihoods[category][label] = {}
        d = self.tokenFreqGivenLabel[category][label]
        self.vocabGivenLabel[category][label] += len(tokens)
        self.labelFreq[category][label] += 1.0
        for i in tokens:
            d[i] = 1.0 + d.get(i, 0.0)
            self.likelihoods[category][label][i] = d[i] / self.vocabGivenLabel[category][label]
        self.trainingSize += 1.0
        self.priors[category][label] = self.labelFreq[category][label] / self.trainingSize

    def train(self, inputFile):
        t = time.time()
        with file(inputFile) as f:
            cnt = 0
            for line in f:
                featureVector, label = self.parse(line)
                self.update(featureVector, label)
                cnt += 1
                if not cnt % 100000:
                    print "Records processed:", cnt
        print "Training done in %f seconds" % (time.time() - t)

    def classify(self, inputVector, actual):
        category, tokens = inputVector
        def _posterior(label):
            posterior = self.priors[category][label]
            for i in tokens:
                if i in self.likelihoods[category][label]:
                    p = self.likelihoods[category][label][i]
                else:
                    p = 1.0 / self.trainingSize
                posterior *= p
            return posterior
        posteriors = {}
        for label in self.likelihoods.get(category, {}):
            posteriors[label] = _posterior(label)
        if not posteriors:
            return None
        bestPosterior = max(posteriors.values())
        matches = [i for i in posteriors if posteriors[i] == bestPosterior]
        if len(matches) == 1:
            return matches[0]
        return None

if __name__ == '__main__':
    c = NB()
    c.train("train.tsv")

    correct = 0
    wrong = 0
    passed = 0
    with file("cv.tsv") as f:
        i = 0
        for line in f:
            inputVector, label = c.parse(line)
            guessed = c.classify(inputVector, label)
            if not guessed:
                passed += 1
            elif guessed == label:
                correct += 1
            else:
                wrong += 1
            i += 1
            if not i % 10:
                print "i: %d, correct: %d, wrong: %d, passed: %d" % (i, correct, wrong, passed)
            if i == 1000:
                break
        print "Correct guesses:", correct
        print "Wrong guesses:", wrong
        print "Passed:", passed
        if correct:
            print "Accuracy:", correct * 100.0 / (correct + wrong)
