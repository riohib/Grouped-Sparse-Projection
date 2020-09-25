import torch
import numpy as np
from numpy import linalg as LA
import pickle
import scipy.io
import logging
import pdb
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sparsity(matrix):
    ni = matrix.shape[0]

    zero_col_ind = (matrix.sum(0) == 0).nonzero().view(-1)  # Get Indices of all zero vector columns.

    # pdb.set_trace()

    spx_c = (np.sqrt(ni) - torch.norm(matrix,1, dim=0) / torch.norm(matrix,2, dim=0)) / (np.sqrt(ni) - 1)

    if len(zero_col_ind) != 0:
        spx_c[zero_col_ind] = 1  # Sparsity = 1 if column already zero vector.

    # pdb.set_trace()
    sps_avg =  spx_c.sum() / matrix.shape[1]

    return sps_avg

def pad_input_dict(in_dict):
    ni_list = [x.shape[0] for x in in_dict.values()]
    max_rows = max(ni_list)

    matrix = torch.zeros(max_rows, len(in_dict))

    for ind in range(len(in_dict)):
        matrix[:ni_list[ind],ind] = in_dict[ind]
    return matrix, ni_list


def checkCritical(pos_matrix, precision=1e-6):
    max_elems = torch.max(pos_matrix, 0)[0]

    ind_crit_bool = (abs(pos_matrix - max_elems) < precision)
    crit_points = pos_matrix * ind_crit_bool

    num_crit_points = torch.sum(ind_crit_bool, dim=0)

    # Boolean of vector cols with non-trivial critical values
    crit_cols = torch.where(num_crit_points.float() > 1, torch.ones(pos_matrix.shape[1], device=device), \
                            torch.zeros(pos_matrix.shape[1], device=device))

    # getting non-trivial critical values
    critval_list = max_elems[crit_cols.bool()]
    critval_all_col = max_elems * crit_cols

    return critval_list, max_elems, critval_all_col


def gmu(p_matrix, xp_mat, mu=0, *args):
    ni_tensor, inv_mask = args

    vgmu = 0
    gradg = 0
    ni_tlist = ni_tensor.int()
    
    p_matrix = torch.abs(p_matrix)
    glist = []

#----------------------------------------------------------------------------------------
    # ni_tensor
    betai = 1 / (torch.sqrt(ni_tensor) - 1)

    xp_mat = p_matrix - (mu * betai)
    indtp = xp_mat > 0

    xp_mat.relu_()


    # outputs
    mnorm = torch.norm(xp_mat, dim=0)
    mnorm_inf = mnorm.clone()
    mnorm_inf[mnorm_inf == 0] = float("Inf")
    col_norm_mask = (mnorm > 0)


    # vgmu calculation
    ## When indtp is not empty (the columns whose norm are not zero)
    xp_mat *= inv_mask 
    xp_mat[:, col_norm_mask] /= mnorm[col_norm_mask]

    ## When indtp IS empty (the columns whose norm ARE zero)
    max_elem_rows = torch.argmax(p_matrix, dim=0)[~col_norm_mask]  # The Row Indices where maximum of that column occurs
    xp_mat[max_elem_rows, ~col_norm_mask] = 1

    # vgmu computation
    vgmu_mat = betai * torch.sum(xp_mat, dim=0)
    vgmu = torch.sum(vgmu_mat)

    return vgmu, xp_mat


# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------- #
def groupedsparseproj(in_dict, sps, precision=1e-6, linrat=0.9):
    # sps = 0.9 ;  precision=1e-6; linrat=0.9
    epsilon = 10e-15
    k = 0
    muup0 = 0

    matrix, ni_list = pad_input_dict(in_dict)
    ni_tensor = torch.tensor(ni_list, device=device, dtype=torch.float32)

    # --------------- Create Mask ---------------------
    inv_mask = torch.zeros(matrix.shape, device=device, dtype=torch.float32)
    for i in range(matrix.shape[1]):
        inv_mask[:ni_list[i],i] = torch.ones(ni_list[i])
    # -------------------------------------------------

    r = matrix.shape[1]  # No of Columns
    critmu = torch.tensor([])
    critval_list = []

    vgmu = torch.zeros(1, device=device)
    # maxxi_list = []

    # These operations were inside the loop, but doesn't need to be.
    matrix_sign = torch.sign(matrix)
    pos_matrix = matrix_sign * matrix
    xp_vec = torch.zeros([matrix.shape[0], matrix.shape[1]]).to(device)
    ni = matrix.shape[0]

#----------------------------------------------------------------------------------------
    k = sum(np.sqrt(ni_list)/(np.sqrt(ni_list)-1))


    # check critical values of mu where g(mu) is discontinuous, that is,
    # where the two (or more) largest entries of x{i} are equal to one another.
    critical_val, max_xi, cval_all_col = checkCritical(pos_matrix)

    muup0 = max(max_xi * (np.sqrt(ni_tensor) - 1))

    # cval_all_col was extracted for the sole reason that we can multiply the critical
    # values withe the appropriate column ni below. Hence, it preserves the column information
    # of where the critical values came from.
    critmu = cval_all_col * (np.sqrt(ni_tensor) - 1) 
    critmu = critmu[critmu > 1e-6] # we only need the critival values here, not the zeros in col.

    k = k - r * sps

    # -------------------- gmu --------------------
    xp_mat = torch.zeros([pos_matrix.shape[0], pos_matrix.shape[1]]).to(device)
    # gmu_args = {'xp_mat':xp_mat, 'ni_tensor':ni_tensor}
    
    vgmu, xp_mat = gmu(pos_matrix, xp_mat, 0, ni_tensor, inv_mask)

#----------------------------------------------------------------------------------------


    if vgmu < k:
        xp_mat = matrix
        gxpmu = vgmu
        numiter = 0
        return xp_mat
    else:
        numiter = 0
        mulow = 0
        glow = vgmu
        muup = muup0
        # Initialization on mu using 0, it seems to work best because the
        # slope at zero is rather steep while it is gets falt for large mu
        newmu = 0
        gnew = glow
        gpnew = gradg  # g'(0)
        delta = muup - mulow
        switch = True

        while abs(gnew - k) > precision * r and numiter < 100:
            oldmu = newmu
            # % Secant method:
            # % newmu = mulow + (k-glow)*(muup-mulow)/(gup-glow);

            # % Bisection:
            # % newmu = (muup+mulow)/2;
            # % Newton:
            newmu = oldmu + (k - gnew) / (gpnew + epsilon)

            if (newmu >= muup) or (newmu <= mulow):  # If Newton goes out of the interval, use bisection
                newmu = (mulow + muup) / 2

            gnew, xnew, gpnew = gmu(matrix, xp_vec, newmu)

            if gnew < k:
                gup = gnew
                xup = xnew
                muup = newmu
            else:
                glow = gnew
                mulow = xnew
                mulow = newmu

            # Guarantees linear convergence
            if (muup - mulow) > linrat * delta and abs(oldmu - newmu) < (1 - linrat) * delta:
                newmu = (mulow + muup) / 2
                
                gnew, xnew, gpnew = gmu(matrix, xp_vec, newmu)

                if gnew < k:
                    gup = gnew
                    xup = xnew
                    muup = newmu
                else:
                    glow = gnew
                    mulow = xnew
                    mulow = newmu
                numiter += 1
            numiter += 1

            if critmu.shape[0] != 0 and abs(mulow - muup) < abs(newmu) * precision and \
                    min(abs(newmu - critmu)) < precision * newmu:
                print('The objective function is discontinuous around mu^*.')
                xp = xnew
                gxpmu = gnew
        try:
            xp_vec = xnew
            # print(' xp_vec = xnew')
        except:
            scipy.io.savemat('matrix.mat', mdict={'arr': matrix})

        gxpmu = gnew

    # pdb.set_trace()

    alpha = torch.zeros([1, matrix.shape[1]], device=device)
    for i in range(r):
        alpha[0, i] = torch.matmul(xp_mat[:, i], pos_matrix[:, i])
        xp_mat[:, i] = alpha[:, i] * (matrix_sign[:, i] * xp_mat[:, i])

    return xp_vec


# def load_matrix_debug(test_matrix):
#     with open(test_matrix, "rb") as fpA:  # Pickling
#         matrix = pickle.load(fpA)
#     return matrix



def load_matrix_debug(mat_tuple):
    matrix_1, matrix_2, matrix_3, matrix_4 = mat_tuple
    with open(matrix_1, "rb") as fpA:  # Pickling
        matrix_1 = pickle.load(fpA)
    with open(matrix_2, "rb") as fpA:  # Pickling
        matrix_2 = pickle.load(fpA)
    with open(matrix_3, "rb") as fpA:  # Pickling
        matrix_3 = pickle.load(fpA)
    with open(matrix_4, "rb") as fpA:  # Pickling
        matrix_4 = pickle.load(fpA)

    matrix_1 = torch.from_numpy(matrix_1).view(-1)
    matrix_2 = torch.from_numpy(matrix_2).view(-1)
    matrix_3 = torch.from_numpy(matrix_3).view(-1)
    matrix_4 = torch.from_numpy(matrix_4).view(-1)

    matrix = {0:matrix_1, 1:matrix_2, 2:matrix_3, 3:matrix_4}
    return matrix


# ## ********************************************************************************** ##
mat_tuple = ("matrix_1.pkl", "matrix_2.pkl", "matrix_3.pkl", "matrix_4.pkl")
in_dict = load_matrix_debug(mat_tuple)

# ## ********************************************************************************** ##

start_time = time.time()
sps = 0.9
precision = 1e-6
linrat = 0.9
X = groupedsparseproj(in_dict, sps, precision=1e-6, linrat=0.9)
print("--- %s seconds ---" % (time.time() - start_time))

# r = 100
# n = 10000
# k = 0

# ## Data Loacing
# # mu, sigma = 0, 1 # mean and standard deviation
# # x = np.random.normal(mu, sigma, (10000, 100)) * 10

# with open('matnew.pkl', 'rb') as fin:
#     x = pickle.load(fin)
# ## ****************************************

# xPos = np.abs(x)

# for i in range(r):
#     k = k + np.sqrt(n) / (np.sqrt(n) - 1)

# sp = sparsity(x)

# print("The Sparsity of input set of vectors: " + str(sp))

# xp_vec = groupedsparseproj(x, 0.8)

# spNew = sparsity(xp_vec)

# print("The Output Sparsity: " + str(spNew))