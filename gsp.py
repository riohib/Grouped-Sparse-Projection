import numpy as np
from projection import *

X = np.array( [ [1, 2, 14, 9, -14, 9, -1, 5, -11, 7],
                     [8, 2, -6, -13, -24, -13, -6, 1, 4, -11],
                     [-3, -2, 3, -1, -6, 3, 18, -2, -2, -19] ] )

#LinearHoyer(X, r, maxiter, spar, W = None, H = None)

X_hat = LinearHoyer(X, 25, 150, 0.6)

print(X_hat)

