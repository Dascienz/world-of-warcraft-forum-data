#!/usr/bin/env python3

import numpy as np

def cosine_distance(x, y):
    """Function for calculating cosine distance
    between two arrays.
    ----- Args:
            x: np.array
            y: np.array
    ----- Returns:
            (float) cosine distance
    """
    
    a = np.dot(x,y) #numerator
    b = np.sqrt(np.dot(x,x) * np.dot(y,y)) #denominator
    return 1. - (a / b)

def cosine_similarity(iArray, nArrays):
    """Function for calculating cosine similarities
    between a given array and an array of arrays.
    ----- Args:
            x: np.array of shape (1,S)
            y: multi-dimensional array of shape (n,S)
    ----- Returns:
            list() of cosine distances
    """
    
    return [cosine_distance(iArray,x) for x in nArrays]

if __name__=="__main__":
    
    iArray = np.array([5.,0.,2.,0.])
    
    nArrays = np.array([[5.,0.,2.,0.],
                        [1.,1.,1.,1.],
                        [5.,0.,1.,0.]])
    
    cosineDistances = cosine_similarity(iArray,nArrays)
    
    print("\nTest array: {}".format(iArray))
    print("\nList of arrays: {}".format(nArrays))
    print("\nList of cosine distances: {}".format(cosineDistances))
    