# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features = basicFeatureExtractor(datum)

    "*** YOUR CODE HERE ***"
    one_hot_encoding = getConnectedWhiteRegions(datum)
    enhancedFeatures = np.concatenate((features, one_hot_encoding), axis=0)
    return enhancedFeatures

def getConnectedWhiteRegions(datum):
    """
    Method to get number of white connected regions for a given image.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        One-hot encoded vector, the index of set bit representing the number of white connected 
            regions in a given image.
    """
    datum = datum.reshape((DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT))
    datum = samples.Datum(samples.convertToTrinary(datum), DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    img = datum.getPixels()

    visited = [[False for j in range(DIGIT_DATUM_HEIGHT)] for i in range(DIGIT_DATUM_HEIGHT)]
    counter = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] == 0 and not visited[i][j]:
                visited = dfs(img, visited, i, j)
                counter += 1;

    # print datum
    # print counter
    if counter > 3:
        counter = 3
    one_hot_encoding = [0, 0, 0]
    one_hot_encoding[counter-1] = 1

    return one_hot_encoding

def isSafe(img, visited, row, col):
    return (row >= 0 and row < DIGIT_DATUM_WIDTH and
            col >= 0 and col < DIGIT_DATUM_HEIGHT and
            img[row][col] == 0 and not visited[row][col])

def dfs(img, visited, row, col):
    """
    Depth-first search on image to find connected components.

    Args:
        img: 2-dimensional numpy.array representing a single image.
        visited: python list maintaining visited nodes.
    """
    adjRows = [-1, -1, -1, 0, 0, 1, 1, 1]
    adjCols = [-1, 0, 1, -1, 1, -1, 0, 1]
    visited[row][col] = True
    for i in range(8):
        if isSafe(img, visited, row + adjRows[i], col + adjCols[i]):
            visited = dfs(img, visited, row + adjRows[i], col + adjCols[i])
    return visited

def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     t ruth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    import pdb
    pdb.set_trace()
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
