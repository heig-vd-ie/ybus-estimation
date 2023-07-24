import numpy as np
import random
from scipy.optimize import minimize
from auxiliary import discard_data_points, remove_outliers

global xx, yy


def matrix_algorithm(lines_node_data, vbase, sbase, iter_num=1000, tau=5e-2):
    zbase = vbase ** 2 / sbase
    lines_node_data = discard_data_points(line_nodes_data0=lines_node_data)
    p2, q2, c2, v2, v1 = [lines_node_data['pj'], lines_node_data['qj'], lines_node_data['cj'],
                          lines_node_data['vj'], lines_node_data['vi']]
    data_points = len(lines_node_data["pj"])
    f_mat = np.zeros((3, data_points))
    m_mat = np.zeros((1, data_points))
    q_mat = np.zeros((3, 3))
    c_mat = np.zeros((1, 3))
    for t in range(data_points):
        f_mat[0][t] = 1
        f_mat[1][t] = q2[t] / p2[t]
        f_mat[2][t] = - (q2[t] ** 2 + p2[t] ** 2) / (2 * (v2[t] ** 2) * p2[t])
        m_mat[0][t] = (v1[t] ** 2 - v2[t] ** 2) / (2 * p2[t])
    q_mat[0][0] = 2
    q_mat[1][1] = 2
    c_mat[0][2] = -1
    lam = np.zeros(iter_num)
    d_lam = np.zeros(iter_num)
    theta = np.array([])
    for i in range(iter_num - 1):
        theta = np.matmul(np.linalg.inv((np.matmul(f_mat, np.transpose(f_mat)) + lam[i] * q_mat).tolist()),
                          np.transpose(np.matmul(m_mat, np.transpose(f_mat)) - lam[i] * c_mat))
        d_lam[i] = 0.5 * np.matmul(np.matmul(np.transpose(theta), q_mat), theta) + np.matmul(c_mat, theta)
        lam[i + 1] = lam[i] + tau * d_lam[i]
    theta = theta * zbase
    theta[2] = theta[2] * zbase
    return theta, lam


def minimization_algorithm(lines_node_data, vbase, sbase, fil_qu=4, fil_0=0.1, n_tests=501, n_sample=4001, seq="pos",
                           epsilon=0, fil_v=0.1):
    zbase = vbase ** 2 / sbase
    lines_node_data = discard_data_points(line_nodes_data0=lines_node_data, fil_qu=fil_qu, fil_0=fil_0, fil_v=fil_v)
    p2, q2, c2, v2, v1 = [lines_node_data['pj'], lines_node_data['qj'], lines_node_data['cj'],
                          lines_node_data['vj'], lines_node_data['vi']]
    r, x, z, cost = multiple_line_reg(v1, v2, p2, q2, n_tests, n_sample, epsilon=epsilon)
    error_z = np.mean((np.array(z) - np.power(x, 2) - np.power(r, 2)) / (np.power(x, 2) + np.power(r, 2)))
    if seq == "zero":
        a, a_clean = [r, x, z], [r, x, z]
    else:
        a, a_clean = remove_outliers([r, x, z])
    r_clean, x_clean, z_clean = a_clean
    # Return R and X to physical units ([ohm])
    r0 = np.mean(r_clean) * zbase
    x0 = np.mean(x_clean) * zbase
    z0 = np.mean(z_clean) * zbase * zbase
    # Compute standard deviation in Ohm
    std_r = abs(np.std(r_clean))
    std_x = abs(np.std(x_clean))
    return r0, x0, std_r, std_x, z0, error_z, r, x, z


def multiple_line_reg(v1, v2, p2, q2, n_tests, n_sample, epsilon):
    r = []
    x = []
    z = []
    cost = []
    tests = range(0, n_tests)
    for _ in tests:
        it = [random.randint(0, len(v1) - 1) for _ in range(n_sample)]
        rij_res, xij_res, zij_sq_res, cost = line_reg_no_b(v2[it], v1[it], (-p2[it]), (-q2[it]), "SLSQP", 0, epsilon)
        r.append(rij_res)
        x.append(xij_res)
        z.append(zij_sq_res)
    return r, x, z, cost


def line_reg_no_b(vi, vj, psj, qsj, my_method, display, epsilon):
    global xx, yy
    # Dependent variable y = (Vi^2 -Vj^2)/(2*Pj)
    yy = np.divide((np.square(vi) - np.square(vj)), 2 * np.array(psj))
    # Matrix of independent variables XX = [1; -(Pj^2 + Qj^2)/(2*Vi^2*Psj); Qsj/Psj ]
    xx = build_x_dist(vi, psj, qsj)
    mu = np.mean(xx, 0)
    sig = np.std(xx, 0)
    # Normalise columns of XX variables
    xx, yy = feature_normalise(xx, yy, mu, sig)
    # Add columns on ones to XX and extract problem size parameters
    m, n = xx.shape
    n = n + 1
    xx = np.hstack([np.ones((m, 1)), xx])
    # Randomise initial guess in range of eps
    theta0 = np.zeros(n)
    for k in range(0, n):
        theta0[k] = epsilon + random.random() * (1 - epsilon)
    cons = ({'type': 'eq', 'fun': lambda x: np.square(denorm_tht(x, mu, sig)[0])
            + np.square(denorm_tht(x, mu, sig)[2]) - denorm_tht(x, mu, sig)[1]})
    if my_method == 'Newton':
        res = minimize(loss, theta0, method='Newton-CG', jac=jac_orb,
                       options={'xtol': 1e-14, 'disp': False, 'maxiter': 1000})
    else:
        res = minimize(loss, theta0, method='SLSQP', constraints=cons,
                       bounds=[(epsilon, 1) for _ in range(xx.shape[1])],
                       options={'ftol': 1e-10, 'disp': False, 'maxiter': 1000})
    theta_res = denorm_tht(res.x, mu, sig)
    [rij_res, xij_res, zij_sq_res] = compute_rxz_from_theta(theta_res)
    cost = np.sum(np.square(np.dot(xx, res.x) - yy))
    if display:
        print('Result real-unit:  ', theta_res)
        print('Cost function J(theta):', np.sum(np.square(np.dot(xx, res.x) - yy)))
    return rij_res, xij_res, zij_sq_res, cost


def build_x_dist(vi, pj, qj):
    vi = np.array(vi)
    m = vi.shape[0]
    x = np.zeros((m, 2))
    x[0::, 0] = -(np.square(pj) + np.square(qj)) / (2 * np.multiply(np.square(vi), pj))
    x[0::, 1] = np.divide(qj, pj)
    return x


def feature_normalise(x, y, mu, sig):
    if not np.all(sig):
        Exception('There is a feature of ones, I cannot normalise')
    x = np.divide((x - mu), sig)
    return x, y


def denorm_tht(t_n, mu, sig):
    t = np.zeros(t_n.shape)
    t[1::] = np.divide(t_n[1::], sig.T)
    t[0] = t_n[0] - np.sum(np.multiply(t[1::], mu.T))
    return t


def jac_orb(x):
    global xx, yy
    diff = (np.dot(xx, x) - yy)
    dot_prod = np.dot(diff.T, xx)
    return dot_prod.T


def loss(x):
    global xx, yy
    return np.sum(np.square((np.dot(xx, x) - yy)))


def compute_rxz_from_theta(theta):
    r = theta[0]
    z_sq = theta[1]
    x = theta[2]
    return r, x, z_sq
