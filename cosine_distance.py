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
    
    epsilon = 1e-9
    a = np.dot(x,y) #numerator
    b = np.sqrt(np.dot(x,x) * np.dot(y,y)) + epsilon #denominator
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
    
    print("\n")
    print("Test array:\n{}\n".format(iArray))
    print("List of arrays:\n{}\n".format(nArrays))
    print("List of cosine distances:\n{}\n".format(cosineDistances))
    