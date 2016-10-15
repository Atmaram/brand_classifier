import matplotlib
import numpy
import matplotlib.pyplot as plt

def getDataSize(inputFile):
    size = 0
    with file(inputFile) as f:
        for line in f:
            size += 1
    return size

def split(inputFile, size):
    trainingSize = 0.6 * size
    cvSize = 0.2 * size
    with file(inputFile) as f:
        lineCount = 0
        with file("train.tsv", "w") as tf:
            for line in f:
                tf.write(line)
                lineCount += 1
                if lineCount >= trainingSize:
                    break
        with file("cv.tsv", "w") as cvf:
            for line in f:
                cvf.write(line)
                lineCount += 1
                if lineCount >= trainingSize + cvSize:
                    break
        with file("test.tsv", "w") as tf:
            for line in f:
                tf.write(line) 

def unseenLabelCount():
    trainingLabels = set()
    cvLabels = set()
    uniqueLabels = set()
    with file("train.tsv") as f:
        for line in f:
            features = line.split("\t")
            trainingLabels.add(features[1].strip())
    with file("cv.tsv") as f:
        for line in f:
            features = line.split("\t")
            cvLabels.add(features[1].strip())
    print "Total unique labels in training set:", len(trainingLabels)
    print "Total unique labels in CV set:", len(cvLabels)
    print "Total unique labels:", len(trainingLabels.union(cvLabels))
    print "Labels in CV set, not in training set:", len(cvLabels - trainingLabels)

def analyseCategories(fileName):
    categories = {}
    with file(fileName) as f:
        for line in f:
            features = line.split("\t")
            category = features[2].strip()
            brand = features[1].strip()
            if category not in categories:
                categories[category] = set()
            categories[category].add(brand)
    
    x = range(len(categories))
    y = [len(i) for i in categories.values()]
    plt.bar(x, y)
    plt.show()
    return categories

if __name__ == '__main__':
    #split()
    unseenLabelCount()
    trainCat = analyseCategories("train.tsv")
    print "Max brands in a category:", max([len(i) for i in trainCat.values()])
    cvCat = analyseCategories("cv.tsv")
    brandExtensionInstance = 0
    newBrandInstance = 0
    brands = set()
    for i in trainCat.values():
        for b in i:
            brands.add(b)
    for c in cvCat:
        for b in cvCat[c]:
            if (b not in trainCat.get(c, {})):
                if b in brands:
                    brandExtensionInstance += 1
                else:
                    newBrandInstance += 1
    print "Brands extended instances:", brandExtensionInstance
    print "New brand seen instances:", newBrandInstance

