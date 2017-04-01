import numpy as np
import rbm
import projectLib as lib
import linearRegression as lr
import sys
from multiprocessing import Pool

#global variables
training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

# F is the number of hidden units
def executeRBM(F):
    K = 5

    # SET PARAMETERS HERE!!!
    # number of hidden units
    #F = 10
    epochs = 150
    epochs_in_interval = 25

    gradientLearningRate = 0.1  
    l = 0.01

    epoch_count = 1    

    W = rbm.getInitialWeights(trStats["n_movies"], F, K) # no.ofusers * m * F * 5
    posprods = np.zeros(W.shape) # m x F x 5
    negprods = np.zeros(W.shape) # m x F x 5
    
    file = open(str(F) + "results.txt",'w')
    file.write("F = " + str(F) + "l = " + str(l) + " epochs_in_interval = " + str(epochs_in_interval) + \
        "\ngradientLearningRate = " + str(gradientLearningRate) + "\n\n")
    for epoch in range(1, epochs + 1):
        if (epoch_count % epochs_in_interval == 0):
            gradientLearningRate *= 10 ** -1

        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(trStats["u_users"]) # 1787 ratings
        np.random.shuffle(visitingOrder)
    
        try:
            for user in visitingOrder:
                # get the ratings of that user
                # [ [m1_idx, rating], [m2_idx, rating] ]
                ratingsForUser = lib.getRatingsForUser(user, training)
    
                # build the visible input
                v = rbm.getV(ratingsForUser)
    
                # get the weights associated to movies the user has seen
                weightsForUser = W[ratingsForUser[:, 0], :, :] # W.shape = m x F x 5
    
                ### LEARNING ###
                # propagate visible input to hidden units
                posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)
                # get positive gradient
                # note that we only update the movies that this user has seen!
                posprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(v, posHiddenProb)
    
                ### UNLEARNING ###
                # sample from hidden distribution
                sampledHidden = rbm.sample(posHiddenProb)
                # propagate back to get "negative data"
                negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
                # propagate negative data to hidden units
                negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
                # get negative gradient
                # note that we only update the movies that this user has seen!
                negprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(negData, negHiddenProb)
    
                # we average over the number of users
                grad = gradientLearningRate * (posprods - negprods) / trStats["n_users"]
                reg_W = np.zeros(W.shape)
                reg_W[ratingsForUser[:, 0], :, :] = weightsForUser

                W += (grad - (gradientLearningRate * l * reg_W))
    
            # Print the current RMSE for training and validation sets
            # this allows you to control for overfitting e.g
            # We predict over the training set
            tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
            trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)
    
            # We predict over the validation set
            vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
            vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)
    
            print("### EPOCH %d ###" % epoch)
            print("Training loss = %f" % trRMSE)
            print("Validation loss = %f" % vlRMSE)
            file.write(str(epoch) + ",")
            file.write(str(trRMSE) + ",")
            file.write(str(vlRMSE) + "\n")
            file.flush()

        except Exception as e:
            file.write("EXCEPTION RAISED: " + str(e) + "\n")

    file.write("\n")
    file.close()


def main():
 
    # Initialise all our arrays
    with Pool(3) as p:
        p.map(executeRBM,[16 , 32 , 64])


    ### END ###
    # This part you can write on your own
    # you could plot the evolution of the training and validation RMSEs for
    # example
    #predictedRatings = np.array([rbm.predictForUser(user, W, training) for user in trStats["u_users"]])
    #np.savetxt("predictedRatings.txt", predictedRatings)

    ### Linear Regression ###
    file = open("linear_reg_results.txt",'w')
    A = getA(training)
    c = getc(rBar, trStats["ratings"])
    
    for n in [-2,-1,0,1,2]:
        l = 10 ** -n
        b = param_reg(A, c, l)
        file.write("l = " + str(l) + " \n")
        file.write(str(lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])) + "\n\n")

    file.close()


if __name__ == "__main__":
    sys.exit(int(main() or 0))
    