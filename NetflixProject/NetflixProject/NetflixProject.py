import projectLib as pl
import sys
import numpy as np

training_data = pl.getTrainingData() #80%  [[movieID,userID,ratingID], ...[ ] ]
validation_data = pl.getValidationData() #20% predict and check with this data

def main():
    
    #for text in text_array:
    #    print(text) #outputs [movieID,userID,ratingID]

    #pass

    return None

def question1_1():

    #populating A :=  , c = rui - avg r
    #axis = 0 maximise along the columns
    #axis = 1 maximise along the rows
    M = np.ndarray.max(training_data, axis = 0)[0]
    U = np.ndarray.max(training_data, axis = 0)[1]
    A = np.zeros((len(training_data) , M+U))
    for i in range(len(training_data)):
        pair = training_data[i]
        A[i][pair[0]] = 1
        A[i][pair[1]] = 1
    
    ratings = training_data[:,2]
    c = ratings - np.mean(training_data, axis = 0)[2]
    #lr.param(A,c)

        
    


if __name__ == "__main__":
    #sys.exit(int(main() or 0))
    main() 