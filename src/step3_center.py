import numpy as np
from preprocessing import center

X = np.load("data/mixed/X_mixed.npy")
Xc, mean = center(X)

print("Shape before:", X.shape, " after:", Xc.shape)
print("Means before centering:", X.mean(axis=1))
print("Means after centering: ", Xc.mean(axis=1))
print("All means ~ 0?", np.allclose(Xc.mean(axis=1), 0))

np.save("data/mixed/X_centered.npy", Xc)