import numpy as np
import matplotlib.pyplot as plt

BETA = 1
GAMMA = 1 / 3
NU = 1 / 50
T = 200

S = np.zeros(T)
I = np.zeros(T)
R = np.zeros(T)

I[0] = 0.1
S[0] = 1 - I[0] - R[0]

for t in range(1, T):
    dst = -BETA * S[t - 1] * I[t - 1] + NU * R[t - 1]
    dit = BETA * S[t - 1] * I[t - 1] - GAMMA * I[t - 1]
    drt = GAMMA * I[t - 1] - NU * R[t - 1]

    S[t] = S[t - 1] + dst
    I[t] = I[t - 1] + dit
    R[t] = R[t - 1] + drt

# Plot
x = np.linspace(0, 1, T)

plt.figure(figsize=(16, 9))
plt.plot(x, S, 'b', label="Susceptible")
plt.plot(x, I, 'r', label="Infected")
plt.plot(x, R, 'g', label="Recovered")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportion")
plt.title("Proportion of Population at each Stage over Time")

plt.show()
