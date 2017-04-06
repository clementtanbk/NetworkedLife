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
#args = (F, gradientLearningRate, l, epochs_in_interval, multiplier,process)
def executeRBM(args):
    
    multiplier_count = 0

    K = 5
    # SET PARAMETERS HERE!!!
    # number of hidden units
    epochs = 500
    F = args[0]
    epochs_in_interval = args[3]
    multiplier = args[4]
    gradientLearningRate = args[1]
    l = args[2]
    process = args[5] 

    prev_rmse = 100000
    cur_rmse = 100000

    increase_count = 0

    W = rbm.getInitialWeights(trStats["n_movies"], F, K) # no.ofusers * m * F * 5
    posprods = np.zeros(W.shape) # m x F x 5
    negprods = np.zeros(W.shape) # m x F x 5
    
    file = open(str(process) + " _"+ str(F) + "results.txt",'w')
    file.write("F = " + str(F) + " l = " + str(l) + " epochs_in_interval = " + str(epochs_in_interval) + \
        "\ngradientLearningRate = " + str(gradientLearningRate) + "\n\n")

    bestW = W

    for epoch in range(1, epochs + 1):


        if (epoch % epochs_in_interval == 0):
            gradientLearningRate *= 10 ** -1
            gradientLearningRate = round(gradientLearningRate,10)

            if(epochs_in_interval < 30):
                epochs_in_interval *= multiplier
                multiplier_count += 1
            #else:
            #    epochs_in_interval += epochs_in_interval

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
                #if( F == 16 and user == 32 or F == 16 and user == 4):
                #    print("user : %s" %user)
                #    print(len(weightsForUser))
                #    print(len(ratingsForUser))

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

                #ADDED REGULARIZATION#
                reg_W = np.zeros(W.shape)
                reg_W[ratingsForUser[:, 0], :, :] = weightsForUser

                #if( F == 16 and user == 32):
                     #print(user)
                     #print(weightsForUser)
               
                W += (grad - (gradientLearningRate * l * reg_W))
    
            # Print the current RMSE for training and validation sets
            # this allows you to control for overfitting e.g
            # We predict over the training set
            tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
            trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)
    
            # We predict over the validation set
            vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
            vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)
    
            print("### F = %s EPOCH %d ###" % (F,epoch))
            print("Gradient Learning Rate: %s " %gradientLearningRate)
            print("Epochs in interval: %s " %epochs_in_interval)
            print("Training loss = %f" % trRMSE)
            print("Validation loss = %f" % vlRMSE)
            file.write(str(epoch) + ",")
            file.write(str(trRMSE) + ",")
            file.write(str(vlRMSE) + ",")
            file.write(str(gradientLearningRate) + ",")
            file.write(str(epochs_in_interval) + "\n")
            file.flush()

            cur_rmse = vlRMSE
            
            if(cur_rmse - prev_rmse > 0.007):
                increase_count += 2

            elif(cur_rmse > prev_rmse):
                increase_count += 1

            else:
                bestW = W

            #if (increase_count >= 2):
            #    increase_count = 0
            #    gradientLearningRate *= 10 ** -1
            #    #gradientLearningRate = round(gradientLearningRate,10)

            prev_rmse = cur_rmse

            if(epoch % 100 == 0):
                predictedRatings = np.array([rbm.predictForUser(user, bestW, training) for user in trStats["u_users"]])
                np.savetxt(str(process) + " _" "predictedRatings.txt", predictedRatings)

        except Exception as e:
            file.write("EXCEPTION RAISED: " + str(e) + "\n")


    file.write("\n")
    file.close()



    #predict of test data and write results to file



def main():
    
    file = open("linear_reg_results.txt",'w')
    rBar = np.mean(trStats["ratings"])
    A = lr.getA(training)
    c = lr.getc(rBar, trStats["ratings"])
    
    for n in [-2,-1,0,1,2]:
        l = 10 ** -n
        b = lr.param_reg(A, c, l)
        file.write("l = " + str(l) + " \n")
        file.write(str(lib.rmse(lr.predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])) + "\n\n")

    file.close()

    # Process on 3 cores for the 3 executions
    with Pool(3) as p:
        # executeRBM(F, gradientLearningRate, l, epochs_in_interval, multiplier, process):
        p.map(executeRBM,[(16 , 0.01, 0.001, 5, 2,1) , 
                          (16 , 0.01, 0.0005, 5, 3,2),
                          (16 , 0.005, 0.0005, 5, 3,3)])


    ### END ###
    # This part you can write on your own
    # you could plot the evolution of the training and validation RMSEs for
    # example

    ### Linear Regression ###



if __name__ == "__main__":
    sys.exit(int(main() or 0))
    
