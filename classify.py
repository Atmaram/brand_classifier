from abc import ABCMeta, abstractmethod
import time
import re

class Classifier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, inputFile):
        return NotImplemented

    @abstractmethod
    def classify(self, inputVector, *args):
        return NotImplemented

    def tokenize(self, text):
        stopWords = set(['as', 'at', 'for', 'is',
                ])

        def _strip(w):
            return w.lower().replace('"'," ").strip()

        def _doSkip(w):
            if not w:
                return True
            if w in stopWords:
                return True
            try:
                i = int(w)
                return True
            except:
                return False
        #text = re.sub("[-/(),.]", " ", text, re.DOTALL).lower()
        text = _strip(text)
        #tokens = [i for i in text.split() if not _doSkip(i)]
        return [_strip(i) for i in text.split()]
        ngrams = tokens[:1] + [" ".join(i) for i in zip(tokens[1:], tokens[2:])]
        return ngrams

    def parse(self, inputLine):
        rawFeatures = inputLine.strip().split("\t")
        return (rawFeatures[2].strip(), self.tokenize(rawFeatures[0])), rawFeatures[1].strip()

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
        self.vocab = {}

    def update(self, featureVector, label):
        category, tokens = featureVector
        if not tokens:
            return
        if category not in self.categoryFilter:
            self.categoryFilter[category] = set()
        self.categoryFilter[category].add(label)
        if category not in self.tokenFreqGivenLabel:
            self.tokenFreqGivenLabel[category] = {}
            self.vocabGivenLabel[category] = {}
            self.labelFreq[category] = {}
            self.likelihoods[category] = {}
            self.priors[category] = {}
            self.vocab[category] = {}

        if tokens[0] not in self.tokenFreqGivenLabel[category]:
            self.tokenFreqGivenLabel[category][tokens[0]] = {}
            self.vocabGivenLabel[category][tokens[0]] = {}
            self.labelFreq[category][tokens[0]] = {}
            self.likelihoods[category][tokens[0]] = {}
            self.vocab[category][tokens[0]] = set()
            self.priors[category][tokens[0]] = {}

        if label not in self.tokenFreqGivenLabel[category][tokens[0]]:
            self.tokenFreqGivenLabel[category][tokens[0]][label] = {}
            self.vocabGivenLabel[category][tokens[0]][label] = 0
            self.labelFreq[category][tokens[0]][label] = 0
            self.likelihoods[category][tokens[0]][label] = {}

        d = self.tokenFreqGivenLabel[category][tokens[0]][label]
        self.vocabGivenLabel[category][tokens[0]][label] += len(tokens) - 1
        self.labelFreq[category][tokens[0]][label] += 1.0
        for i in tokens[1:]:
            d[i] = 1.0 + d.get(i, 0.0)
            self.likelihoods[category][tokens[0]][label][i] = d[i] / self.vocabGivenLabel[category][tokens[0]][label]
            self.vocab[category][tokens[0]].add(i)
        self.trainingSize += 1.0
        self.priors[category][tokens[0]][label] = self.labelFreq[category][tokens[0]][label] / self.trainingSize

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

    def posterior(self, category, label, tokens):
        posterior = self.priors[category][tokens[0]][label]
        for i in tokens[1:]:
            if i not in self.vocab[category][tokens[0]]:
                continue
            if i in self.likelihoods[category][tokens[0]][label]:
                p = self.likelihoods[category][tokens[0]][label][i]
            else:
                p = 1.0 / self.trainingSize
            posterior *= p
        return posterior

    def classify(self, inputVector):
        category, tokens = inputVector
        if not tokens:
            return None
        posteriors = {}
        for label in self.likelihoods.get(category, {}).get(tokens[0], {}):
            posteriors[label] = self.posterior(category, label, tokens)
        if not posteriors:
            return None
        bestPosterior = max(posteriors.values())
        #den = sum(posteriors.values())
        #if not den:
        #    return None
        matches = [i for i in posteriors if posteriors[i] == bestPosterior]
        #matches = [i for i in posteriors if (posteriors[i] / den) > 0.99]
        if len(matches) == 1:
            return matches[0]
        return None

    def debugDump(self, inputVector, guessed, actual):
        category, tokens = inputVector
        if not tokens:
            return
        with file("debug.csv", "w") as f:
            header = ['Brand', 'Posterior']
            gRow = [guessed, self.posterior(category, guessed, tokens)]
            aRow = [actual, self.posterior(category, actual, tokens)]
            for token in tokens[1:]:
                lg = self.likelihoods[category][tokens[0]][guessed].get(token, 0)
                la = self.likelihoods[category][tokens[0]][actual].get(token, 0)
                if lg > la:
                    header.append(token)
                    gRow.append(lg)
                    aRow.append(la)
            f.write(",".join([i for i in header]) + "\n")
            f.write(",".join([str(i) for i in gRow]) + "\n")
            f.write(",".join([str(i) for i in aRow]) + "\n")


if __name__ == '__main__':
    import sys
    c = NB()
    c.train("train.tsv")

    def _evaluate(fileName, debug=False):
        correct = 0
        wrong = 0
        passed = 0
        with file(fileName) as f:
            i = 0
            for line in f:
                inputVector, label = c.parse(line)
                guessed = c.classify(inputVector)
                if not guessed:
                    passed += 1
                elif guessed == label:
                    correct += 1
                else:
                    wrong += 1
                    if debug:
                       if wrong == 1:
                        print inputVector
                        c.debugDump(inputVector, guessed, label)
                        break
                i += 1
                if not i % 10000:
                    print "i: %d, correct: %d, wrong: %d, passed: %d" % (i, correct, wrong, passed)
            print "validation results of", fileName
            print "================================"
            print "Correct guesses:", correct
            print "Wrong guesses:", wrong
            print "Passed:", passed
            if correct:
                print "Accuracy:", correct * 100.0 / (correct + wrong)

    _evaluate("cv.tsv", debug=len(sys.argv) > 1)
    c.train("cv.tsv")
    _evaluate("test.tsv")
