<<<<<<< HEAD:Assignment 2/A2.py
# for most part, the formula is found in the textbook

import numpy as np
import numpy.linalg as la
from itertools import product


def clean(num):
    # Bounds rating between 1 and 5
    if num > 5:
        return 5
    elif num < 1:
        return 1
    return num


def MSE(arr1, arr2):
    # Calculates the mean square error
    return np.nanmean((arr1 - arr2) ** 2) ** 0.5


R = np.array(
    np.matrix(
        "5 0 5 4; "
        "0 1 1 4; "
        "4 1 2 4; "
        "3 4 0 3; "
        "1 5 3 0"), dtype=np.float)

for r in R:
    for i in range(len(r)):
        if r[i] == 0:
            r[i] = np.NaN


N_Users, N_Movies = R.shape
N_datapoints = N_Users * N_Movies - np.isnan(R).sum()
MEAN_RATING = np.nanmean(R)

A = np.zeros((N_datapoints, N_Users + N_Movies))
c = np.zeros((N_datapoints, 1))
labels = np.zeros((N_datapoints, 1))

# Initialize C vector
i = 0
for r in R.ravel():
    if not np.isnan(r):
        c[i] = r - MEAN_RATING
        i += 1

# Initialize A matrix
k = 0
for i in range(len(R)): # loop through rows first, which represents users
    r = R[i]
    for j in range(len(r)): # loop through movie next
        if not np.isnan(r[j]):
            # Set A[row, column (user and movie) ] to 1
            A[k, i] = 1
            A[k, N_Users + j] = 1
            k += 1

# Solve for b (users + movies concatenated)
b = la.lstsq(A, c)[0]
print(b)

k = 0
rHat_1 = np.zeros_like(R)
preds = A.dot(b) + MEAN_RATING
for i, j in product(range(N_Users), range(N_Movies)):
    # flatten the predictions to the same shape as the matrix R
    if np.isnan(R[i, j]):
        rHat_1[i, j] = np.nan
    else:
        rHat_1[i, j] = clean(preds[k])
        k += 1 

print("Part 1")
print("MSE: %.3f" % MSE(rHat_1, R), end='\n\n')

# Part 2

rTilde = R - rHat_1  # Initalize rTilde

D = np.zeros((N_Movies, N_Movies))  # Initialize Distance matrix to all 0
for i, j in product(range(N_Movies), range(N_Movies)):
    if i == j:
        continue  # if movie i == movie j, distance is 0
    
    top = np.nanmean(rTilde[:, i] * rTilde[:, j])  # numerator

    # Need a mask to compare the corresponding denominator. Example, movie A and movie C rated by users 1 and 4 only
    # So for denominator, withinithe same movie vector, we only take rows 1 and 4. Example. Movie C has 5 non-nan values
    # but we take only row 1 and 4. This is the purpose of the mask
    mask = {-1 if np.isnan(l) else k for k, l in enumerate(rTilde[:, i] * rTilde[:, j])}
    mask.discard(-1)
    mask = tuple(mask)
    bottom = np.sqrt(np.nanmean(rTilde[mask, i] ** 2) * np.nanmean(rTilde[mask, j] ** 2))  # denominator
    
    D[i, j] = top / bottom  # cosine distance


rHat_2 = np.zeros_like(R)
for i, j in product(range(N_Users), range(N_Movies)):
    if np.isnan(R[i, j]):
        rHat_2[i, j] = np.nan
        continue
    value = rHat_1[i, j]  # baseline prediction value
    
    # get the 2 movies closest to the current movie j
    second_movie, top_movie = np.abs(D[j]).argsort()[-2:]  # this is done for clarity

    top, bottom = 0, 0

    # get the numerator and denominator only if user i rated movies k as well
    for k in (second_movie, top_movie):
        if not np.isnan(rTilde[i, k]):
            top += rTilde[i, k] * D[j, k]
            bottom += np.abs(D[j, k])
    
    if bottom > 0:
        value += top / bottom
    
    rHat_2[i, j] = clean(value)

print("Part 2")
print("Neighbours: \n", D)
print("MSE: %.3f" % MSE(rHat_2, R), end='\n\n')


# Part 3

A = np.array([
    [1, 0, 2],
    [1, 1, 0],
    [0, 2, 1],
    [2, 1, 1]
])

c = np.array([2, 1, 1, 3])
b = la.lstsq(A, c)[0]

print("Part 3")
print("Without Regularization")
print("Standard Formulation, b = %s \tMSE: %.5f" % (str(b), MSE(A.dot(b), c)))

# Alternative solution to least square without regularization
b1 = la.inv(A.T.dot(A)).dot(A.T).dot(c)
print("Alternative Formuation, b = %s \tMSE: %.5f" % (str(b1), MSE(A.dot(b1), c)), end="\n\n")

# Solution to ridge regression

size = A.T.dot(A).shape[0]
results = []

print("With Regularization")
for l in np.arange(0, 5.1, 0.2):
    # l refers to the hyper parameter lambda

    b2 = la.inv(A.T.dot(A) + l * np.identity(size)).dot(A.T).dot(c)
    rmse = MSE(A.dot(b2), c)
    print("Lambda = %.1f, b = %s \tMSE: %.4f" % (l, str(b2), rmse))
    results.append((l, rmse))
=======
# for most part, the formula is found in the textbook

import numpy as np
import numpy.linalg as la
from itertools import product


def clean(num):
    # Bounds rating between 1 and 5
    if num > 5:
        return 5
    elif num < 1:
        return 1
    return num


def MSE(arr1, arr2):
    # Calculates the mean square error
    return np.nanmean((arr1 - arr2) ** 2) ** 0.5


R = np.array(
    np.matrix(
        "5 0 5 4; "
        "0 1 1 4; "
        "4 1 2 4; "
        "3 4 0 3; "
        "1 5 3 0"), dtype=np.float)

for r in R:
    for i in range(len(r)):
        if r[i] == 0:
            r[i] = np.NaN


N_Users, N_Movies = R.shape
N_datapoints = N_Users * N_Movies - np.isnan(R).sum()
MEAN_RATING = np.nanmean(R)

A = np.zeros((N_datapoints, N_Users + N_Movies))
c = np.zeros((N_datapoints, 1))
labels = np.zeros((N_datapoints, 1))

# Initialize C vector
i = 0
for r in R.ravel():
    if not np.isnan(r):
        c[i] = r - MEAN_RATING
        i += 1

# Initialize A matrix
k = 0
for i in range(len(R)): # loop through rows first, which represents users
    r = R[i]
    for j in range(len(r)): # loop through movie next
        if not np.isnan(r[j]):
            # Set A[row, column (user and movie) ] to 1
            A[k, i] = 1
            A[k, N_Users + j] = 1
            k += 1

# Solve for b (users + movies concatenated)
b = la.lstsq(A, c)[0]

k = 0
rHat_1 = np.zeros_like(R)
preds = A.dot(b) + MEAN_RATING
for i, j in product(range(N_Users), range(N_Movies)):
    # flatten the predictions to the same shape as the matrix R
    if np.isnan(R[i, j]):
        rHat_1[i, j] = np.nan
    else:
        rHat_1[i, j] = clean(preds[k])
        k += 1 

print("Part 1")
print("MSE: %.3f" % MSE(rHat_1, R), end='\n\n')

# Part 2

rTilde = R - rHat_1  # Initalize rTilde

D = np.zeros((N_Movies, N_Movies))  # Initialize Distance matrix to all 0
for i, j in product(range(N_Movies), range(N_Movies)):
    if i == j:
        continue  # if movie i == movie j, distance is 0
    
    top = np.nanmean(rTilde[:, i] * rTilde[:, j])  # numerator

    # Need a mask to compare the corresponding denominator. Example, movie A and movie C rated by users 1 and 4 only
    # So for denominator, withinithe same movie vector, we only take rows 1 and 4. Example. Movie C has 5 non-nan values
    # but we take only row 1 and 4. This is the purpose of the mask
    mask = {-1 if np.isnan(l) else k for k, l in enumerate(rTilde[:, i] * rTilde[:, j])}
    mask.discard(-1)
    mask = tuple(mask)
    bottom = np.sqrt(np.nanmean(rTilde[mask, i] ** 2) * np.nanmean(rTilde[mask, j] ** 2))  # denominator
    
    D[i, j] = top / bottom  # cosine distance


rHat_2 = np.zeros_like(R)
for i, j in product(range(N_Users), range(N_Movies)):
    if np.isnan(R[i, j]):
        rHat_2[i, j] = np.nan
        continue
    value = rHat_1[i, j]  # baseline prediction value
    
    # get the 2 movies closest to the current movie j
    second_movie, top_movie = np.abs(D[j]).argsort()[-2:]  # this is done for clarity

    top, bottom = 0, 0

    # get the numerator and denominator only if user i rated movies k as well
    for k in (second_movie, top_movie):
        if not np.isnan(rTilde[i, k]):
            top += rTilde[i, k] * D[j, k]
            bottom += np.abs(D[j, k])
    
    if bottom > 0:
        value += top / bottom
    
    rHat_2[i, j] = clean(value)

print("Part 2")
print("Neighbours: \n", D)
print("MSE: %.3f" % MSE(rHat_2, R), end='\n\n')


# Part 3

A = np.array([
    [1, 0, 2],
    [1, 1, 0],
    [0, 2, 1],
    [2, 1, 1]
])

c = np.array([2, 1, 1, 3])
b = la.lstsq(A, c)[0]

print("Part 3")
print("Without Regularization")
print("Standard Formulation, b = %s \tMSE: %.5f" % (str(b), MSE(A.dot(b), c)))

# Alternative solution to least square without regularization
b1 = la.inv(A.T.dot(A)).dot(A.T).dot(c)
print("Alternative Formuation, b = %s \tMSE: %.5f" % (str(b1), MSE(A.dot(b1), c)), end="\n\n")

# Solution to ridge regression

size = A.T.dot(A).shape[0]
results = []

print("With Regularization")
for l in np.arange(0, 5.1, 0.2):
    # l refers to the hyper parameter lambda

    b2 = la.inv(A.T.dot(A) + l * np.identity(size)).dot(A.T).dot(c)
    rmse = MSE(A.dot(b2), c)
    print("Lambda = %.1f, b = %s \tMSE: %.4f" % (l, str(b2), rmse))
    results.append((l, rmse))
>>>>>>> b7d1a0d6554ce09fa038f93038cd6f6c2631a943:Assignment 2/All.py
