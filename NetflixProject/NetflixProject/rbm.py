import numpy as np
import projectLib as lib
import math
import rbm
#import bigfloat

# set highest rating
K = 5

def softmax(x):
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ratingsPerMovie(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()
    return np.array([[i, movie, len([x for x in training if x[0] == movie])] for i, movie in enumerate(u_movies)])

#validated
def getV(ratingsForUser):
    # ratingsForUser is obtained from the ratings for user library
    # you should return a binary matrix ret of size m x K, where m is the number of movies
    #   that the user has seen. ret[i][k] = 1 if the user
    #   has rated movie ratingsForUser[i, 0] with k stars
    #   otherwise it is 0
    ret = np.zeros((len(ratingsForUser), K)) # no. of ratings x 5
    for i in range(len(ratingsForUser)):
        ret[i, ratingsForUser[i, 1]-1] = 1.0
    return ret

def getInitialWeights(m, F, K):
    # m is the number of visible units
    # F is the number of hidden units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, F, K))

def sig(x):
    ### TO IMPLEMENT ###
    # x is a real vector of size n
    # ret should be a vector of size n where ret_i = sigmoid(x_i)
    result = np.zeros(len(x))
    try:
        result = 1.0/(1+np.exp(-1*x))

    #overflow error: math range error at epoch 357
    except RuntimeError:
        print(x)

    except OverflowError:
       print("overflow error, catching exception")

       for i in range(len(x)):
           result[i] = 1.0/(1 + (math.exp(702)))

    return result

#for each user
def visibleToHiddenVec(v, w):
    ### TO IMPLEMENT ###
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # ret should be a vector of size F
    h_intm = np.tensordot(w.swapaxes(0,1), v, axes=([1,2],[0,1]))
    #print(h_intm)
    h = sig(h_intm)

    return h

#for each user
def hiddenToVisible(h, w):
    ### TO IMPLEMENT ###
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # ret should be a matrix of size m x 5, where m
    #   is the number of movies the user has seen.
    #   Remember that we do not reconstruct movies that the user
    #   has not rated! (where reconstructing means getting a distribution
    #   over possible ratings).
    #   We only do so when we predict the rating a user would have given to a movie.
    # w.shape = (M * F * 5)
    v = np.zeros((len(w),len(w[0,0,:]))) # m * 5
    for i in range(len(w)):
        w_i = w[i,:,:] # F * 5
        k_values = np.tensordot(w_i.swapaxes(0,-1), h , axes = (1,0))
        v_i = softmax(k_values)
        #num = np.exp(k_values) # 5 * 1
        #den = np.sum(num)
        #v_i = num / den
        v[i] = v_i
        #print(v)

    #v here is no longer binary! it is a probability distribution of having rated in that kth slot
    #of that v_i vector
    return v

#a = np.array([[[1,1],[1,2]], 
#              [[2,3],[2,2]]])
#b = np.array([1,2])
#print(hiddenToVisible(b,a))

#for each user
def probProduct(v, p):
    # v is a matrix of size m x 5
    # p is a vector of size F, activation of the hidden units
    # returns the gradient for visible input v and hidden activations p
    ret = np.zeros((v.shape[0], p.size, v.shape[1])) # m x F x 5
    for i in range(v.shape[0]):
        for j in range(p.size):
            for k in range(v.shape[1]):
                ret[i, j, k] = v[i, k] * p[j]
    return ret

def sample(p):
    # p is a vector of real numbers between 0 and 1, P(h_j = 1|v) vector
    # ret is a vector of same size as p, where ret_i = Ber(p_i)
    # In other word we sample from a Bernouilli distribution with
    # parameter p_i to obtain ret_i
    samples = np.random.random(p.size)
    return np.array(samples <= p, dtype=int)

def getPredictedDistribution(v, w, wq):
    # W over here has already been updated.

    ### TO IMPLEMENT ###
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq

    #learning
    v_h = visibleToHiddenVec(v,w) #calculates probability of h_j being activated
    pg = probProduct(v,v_h)
    
    #unlearning
    sample_h = sample(v_h)
    h_v = hiddenToVisible(sample_h,wq[np.newaxis,:,:]) #negative data
    #print("h_v %s" %h_v)
    #ng = visibleToHiddenVec(h_v,wq[np.newaxis,:,:])



    # ret is a vector of size 5
    return h_v[0,:]

#w = np.array([[[1,1],[1,2],[3,4]], 
#               [[2,3],[2,2],[1,2]]])
#v = np.array([[1,0],[1,0]])
#wq = w[0,: ,:]
#print(getPredictedDistribution(v,w,wq))

def predictRatingMax(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of two you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the one with the highest probability
    max_rating = np.argmax(ratingDistribution) + 1

    return None

def predictRatingExp(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of two you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation of the ratingDistribution
    mean = 0
    for i in range(len(ratingDistribution)):
        mean += ratingDistribution[i] * (i+1)
    return mean

def predictMovieForUser(q, user, W, training, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    ratingsForUser = lib.getRatingsForUser(user, training)
    v = getV(ratingsForUser)
    ratingDistribution = getPredictedDistribution(v, W[ratingsForUser[:, 0], :, :], W[q, :, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)

def predict(movies, users, W, training, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUser(movie, user, W, training, predictType=predictType) for (movie, user) in zip(movies, users)]

def predictForUser(user, W, training, predictType="exp"):
    ### TO IMPLEMENT
    # given a user ID, predicts all movie ratings for the user
    movies =[]
    for movie in range(len(W)):
        movies.append(predictMovieForUser(movie,user,W,training,predictType))

    return movies