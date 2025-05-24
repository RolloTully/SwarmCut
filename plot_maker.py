import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def monoExp(x, m, t, b):
    return m * np.exp(t * x) + b


x_E591 = np.array([200, 200, 400, 100, 300])
y_E591 = np.array([1181, 1111, 4762, 475, 2344]) # Time to completion

x_S1223 = np.array([200,300,300, 400, 100]) # Number of points
y_S1223 = np.array([1296, 2300, 2429, 4565, 493])

p0 = (249, .001, 400)
params, cv = scipy.optimize.curve_fit(monoExp, np.concatenate([x_E591, x_S1223]), np.concatenate([y_E591, y_S1223]),p0)
m, t, b = params

squaredDiffs = np.square(np.concatenate([y_E591, y_S1223]) - monoExp(np.concatenate([x_E591, x_S1223]), m, t, b))
squaredDiffsFromMean = np.square(np.concatenate([y_E591, y_S1223]) - np.mean(np.concatenate([y_E591, y_S1223])))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

print(rSquared)

y = monoExp(np.linspace(0,400,100), m,t,b)
plt.rcParams["font.family"] = "Times New Roman"
plt.plot(np.linspace(0,400,100), y, label = str(round(m,2))+"$e^{"+str(round(t,4))+"x}+"+str(round(b,2))+"$,$R^{2} = $"+str(round(rSquared,4)), c = "black")
plt.scatter(x_E591,y_E591, label = "E591", marker = "2", c = "black")
plt.scatter(x_S1223,y_S1223, label = "S1223", marker = ".", c = "black")
plt.legend()
plt.xlabel("Discrete points")
plt.ylabel("Time to completion (s)")
plt.show()
