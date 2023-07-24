import numpy as np
import math
import matplotlib as mpl
from typing import TypedDict
import matplotlib.pyplot as plt
from cmath import rect


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def make_bar(ax, x0=0, y0=0, width=0.5, height=1, cmap="viridis", norm=mpl.colors.Normalize(vmin=0, vmax=1), **kwargs):
    # Make data
    u = np.linspace(0, 2 * np.pi, 4 + 1) + np.pi / 4.
    v_ = np.linspace(np.pi / 4., 3. / 4 * np.pi, 100)
    v = np.linspace(0, np.pi, len(v_) + 2)
    v[0] = 0
    v[-1] = np.pi
    v[1:-1] = v_
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    x_thr = np.sin(np.pi / 4.) ** 2
    z_thr = np.sin(np.pi / 4.)
    x[x > x_thr] = x_thr
    x[x < -x_thr] = - x_thr
    y[y > x_thr] = x_thr
    y[y < - x_thr] = - x_thr
    z[z > z_thr] = z_thr
    z[z < - z_thr] = - z_thr

    x *= 1. / x_thr * width
    y *= 1. / x_thr * width
    z += z_thr
    z *= height / (2. * z_thr)
    # translate
    x += x0
    y += y0
    # plot
    ax.plot_surface(x, y, z, cmap=cmap, norm=norm, **kwargs)


def make_bars(ax, x, y, height, cmap, width=1, max_h=10):
    widths = np.array(width) * np.ones_like(x)
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    h = np.array(height).flatten()
    w = np.array(widths).flatten()
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max([max_h, h.max(initial=0)]))
    for i in range(len(x.flatten())):
        make_bar(ax, x0=x[i], y0=y[i], width=w[i], height=h[i], norm=norm, cmap=cmap)


def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list


def ybus_calculation(node_a, node_b, r_re, x_re):
    nodes = sorted(unique(node_a + node_b))
    n = len(nodes)
    incidence = np.zeros((n, n))
    ybus = np.zeros((n, n), dtype=complex)
    for idx, n2 in enumerate(node_b):
        n1 = node_a[idx]
        r = r_re[idx]
        x = x_re[idx]
        y = 1 / (r + 1j * x)
        n1_idx = nodes.index(n1)
        n2_idx = nodes.index(n2)
        incidence[n2_idx, n2_idx] = 1
        incidence[n2_idx, n1_idx] = -1
        ybus[n1_idx, n2_idx] = -y
        ybus[n2_idx, n1_idx] = -y
    ybus = np.diag(-ybus.sum(axis=1)) + ybus
    return incidence, ybus, nodes


class LineNodesData(TypedDict):
    pj: np.ndarray
    qj: np.ndarray
    cj: np.ndarray
    vj: np.ndarray
    vi: np.ndarray
    di: np.ndarray


def remove_data(index, pj, qj, cj, vj, vi):
    if index.size > 0:
        rows = index[0]
        pj = np.delete(pj, rows)
        qj = np.delete(qj, rows)
        cj = np.delete(cj, rows)
        vj = np.delete(vj, rows)
        vi = np.delete(vi, rows)
    return pj, qj, cj, vj, vi


def discard_data_points(line_nodes_data0: LineNodesData, fil_qu: float = 4, fil_0: float = 0.1, fil_v: float = 0.1) \
        -> LineNodesData:
    pj0 = np.array(line_nodes_data0["pj"])
    qj0 = np.array(line_nodes_data0["qj"])
    cj0 = np.array(line_nodes_data0["cj"])
    vj0 = np.array(line_nodes_data0["vj"])
    vi0 = np.array(line_nodes_data0["vi"])
    index_p1 = np.where(abs(pj0 - np.mean(pj0)) > fil_qu * np.std(pj0))
    index_q1 = np.where(abs(qj0 - np.mean(qj0)) > fil_qu * np.std(qj0))
    index_p2 = np.where(abs(pj0) < fil_0 * max(abs(pj0)))
    index_v2 = np.where(abs(vj0) < fil_v * max(abs(vj0)))
    index_p3 = np.where(np.isnan(pj0))
    index_q2 = np.where(np.isnan(qj0))
    index_vi = np.where(np.isnan(vi0))
    index_vj = np.where(np.isnan(vj0))
    index_cj = np.where(np.isnan(cj0))
    index0 = np.concatenate((index_p1, index_q1, index_p2, index_p3, index_q2,
                             index_vi, index_vj, index_vj, index_cj, index_v2), axis=1)
    pj1, qj1, cj1, vj1, vi1 = remove_data(index0, pj0, qj0, cj0, vj0, vi0)
    line_nodes_data_o: LineNodesData = {"pj": pj1, "qj": qj1, "cj": cj1, "vj": vj1, "vi": vi1, "di": np.nan}
    return line_nodes_data_o


def aggregate_data(databases: dict, node_a0, node_b0, df, vbase, sbase):
    v1 = []
    v2 = []
    p2 = []
    q2 = []
    c2 = []
    for i in databases.keys():
        for phase in databases[i]['phase']:
            v11 = df[i][node_a0 + ' U ' + phase + ' [V]']
            v22 = df[i][node_b0 + ' U ' + phase + ' [V]']
            p22 = - df[i][node_b0 + ' P ' + phase + ' [W]']
            q22 = - df[i][node_b0 + ' Q ' + phase + ' [var]']
            c22 = df[i][node_b0 + ' I ' + phase + ' [A]']
            v11 = np.asarray(v11) / vbase
            v22 = np.asarray(v22) / vbase
            p22 = np.asarray(p22) / sbase
            q22 = np.asarray(q22) / sbase
            c22 = np.asarray(c22) * vbase / sbase
            v1.extend(v11)
            v2.extend(v22)
            p2.extend(p22)
            q2.extend(q22)
            c2.extend(c22)
    aggregated_data: LineNodesData = {"pj": np.array(p2), "qj": np.array(q2), "vj": np.array(v2),
                                      "cj": np.array(c2), "vi": np.array(v1), "di": np.nan}
    return aggregated_data


def aggregate_sequences(databases: dict, node_a0, node_b0, df, vbase, sbase, seq="pos"):
    np_rect = np.vectorize(rect)
    a = np_rect(1, np.deg2rad(120))
    v1 = []
    v2 = []
    p2 = []
    q2 = []
    c2 = []
    for i in databases.keys():
        v2a, v2b, v2c = df[i][node_b0 + ' U L1 [V]'], df[i][node_b0 + ' U L2 [V]'], df[i][node_b0 + ' U L3 [V]']
        v1a, v1b, v1c = df[i][node_a0 + ' U L1 [V]'], df[i][node_a0 + ' U L2 [V]'], df[i][node_a0 + ' U L3 [V]']
        p2a, p2b, p2c = df[i][node_b0 + ' P L1 [W]'], df[i][node_b0 + ' P L2 [W]'], df[i][node_b0 + ' P L3 [W]']
        q2a, q2b, q2c = df[i][node_b0 + ' Q L1 [var]'], df[i][node_b0 + ' Q L2 [var]'], df[i][node_b0 + ' Q L3 [var]']
        v20, v21, v22 = (v2a + v2b * (a**2) + v2c * a) / 3, (v2a + v2b + v2c) / 3, (v2a + v2b * a + v2c * (a**2)) / 3
        v10, v11, v12 = (v1a + v1b * (a**2) + v1c * a) / 3, (v1a + v1b + v1c) / 3, (v1a + v1b * a + v1c * (a**2)) / 3
        i2a, i2b, i2c = np.conj((np.array(p2a)+np.array(q2a)*1j) / v2a), \
            np.conj((np.array(p2b)+np.array(q2b)*1j)/(v2b * (a**2))), \
            np.conj((np.array(p2c)+np.array(q2c)*1j)/(v2c * a))
        i20, i21, i22 = (i2a + i2b + i2c) / 3, (i2a + i2b * a + i2c * (a ** 2)) / 3, \
            (i2a + i2b * (a ** 2) + i2c * a) / 3
        p20, p21, p22 = np.multiply(v20, np.conj(i20)).to_numpy().real, np.multiply(v21, np.conj(i21)).to_numpy().real,\
            np.multiply(v22, np.conj(i22)).to_numpy().real
        q20, q21, q22 = np.multiply(v20, np.conj(i20)).to_numpy().imag, np.multiply(v21, np.conj(i21)).to_numpy().imag,\
            np.multiply(v22, np.conj(i22)).to_numpy().imag

        v10, v11, v12 = np.asarray(v10) / vbase, np.asarray(v11) / vbase, np.asarray(v12) / vbase
        v20, v21, v22 = np.asarray(v20) / vbase, np.asarray(v21) / vbase, np.asarray(v22) / vbase
        p20, p21, p22 = - np.asarray(p20) / sbase, - np.asarray(p21) / sbase, - np.asarray(p22) / sbase
        q20, q21, q22 = - np.asarray(q20) / sbase, - np.asarray(q21) / sbase, - np.asarray(q22) / sbase
        i20, i21, i22 = np.asarray(i20) * vbase / sbase, np.asarray(i21) * vbase / sbase, \
            np.asarray(i22) * vbase / sbase

        if seq == "pos":
            v2.extend(v21)
            v1.extend(v11)
            c2.extend(i21)
            p2.extend(p21)
            q2.extend(q21)
        elif seq == "neg":
            v2.extend(v22)
            v1.extend(v12)
            c2.extend(i22)
            p2.extend(p22)
            q2.extend(q22)
        elif seq == "zero":
            v2.extend(v20)
            v1.extend(v10)
            c2.extend(i20)
            p2.extend(p20)
            q2.extend(q20)
    aggregated_data: LineNodesData = {"pj": np.array(np.real(p2)), "qj": np.array(np.real(q2)),
                                      "vj": np.array(np.real(v2)), "cj": np.array(np.real(c2)),
                                      "vi": np.array(np.real(v1)), "di": np.nan}
    return aggregated_data


def synthesize_data(lines_node_data, r: float, x: float, vbase, sbase,
                    vn_std=0.001, pn_std=0.01, qn_std=0.01, cn_std=0.001):
    zbase = vbase ** 2 / sbase
    v2 = lines_node_data["vj"]
    p2 = lines_node_data["pj"]
    q2 = lines_node_data["qj"]
    z2_sq = (r ** 2 + x ** 2)
    c2_sq = (p2 ** 2 + q2 ** 2) / v2 ** 2
    v1_sq = v2 ** 2 + 2 * p2 * r / zbase + 2 * q2 * x / zbase - z2_sq * c2_sq / (zbase ** 2)
    v1 = [math.sqrt(x) for x in v1_sq]
    len_max = v2.size
    v2_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * vn_std) * v2[i] for i in range(len_max)])
    p2_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * qn_std) * p2[i] for i in range(len_max)])
    q2_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * pn_std) * q2[i] for i in range(len_max)])
    c2_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * cn_std) * (c2_sq[i] ** 0.5) for i in range(len_max)])
    v1_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * vn_std) * v1[i] for i in range(len_max)])
    synthesized_data: LineNodesData = {"pj": p2_noisy, "qj": q2_noisy, "vj": v2_noisy, "cj": c2_noisy, "vi": v1_noisy,
                                       "di": np.nan}
    return synthesized_data


def remove_outliers(a):
    mean_a = np.mean(a, axis=1)
    std_a = np.std(a, axis=1)
    a_clean = [x for x in np.array(a).T if (abs(x - mean_a) < std_a).all()]
    a_clean = np.array(a_clean).T.tolist()
    a_clean = [x for x in np.array(a_clean).T if (x > mean_a * 0.1).all()]
    a_clean = np.array(a_clean).T.tolist()
    return a, a_clean


def synthesize_angle(lines_node_data, r: float, x: float, vbase, sbase,
                     vn_std=0.001, pn_std=0.01, qn_std=0.01, cn_std=0.001, dn_std=0.01):
    zbase = vbase ** 2 / sbase
    v2 = lines_node_data["vj"]
    p2 = lines_node_data["pj"]
    q2 = lines_node_data["qj"]
    c2 = ((p2 ** 2 + q2 ** 2) * ((vbase / v2) ** 2)) ** 0.5
    v1_phasor = v2 + (r + 1j * x) * (p2 - 1j * q2) / (v2 * zbase)
    v1 = [np.abs(x) for x in v1_phasor]
    d1 = [np.angle(x, True) for x in v1_phasor]
    len_max = v2.size
    v2_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * vn_std) * v2[i] for i in range(len_max)])
    p2_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * qn_std) * p2[i] for i in range(len_max)])
    q2_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * pn_std) * q2[i] for i in range(len_max)])
    c2_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * cn_std) * c2[i] for i in range(len_max)])
    v1_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * vn_std) * v1[i] for i in range(len_max)])
    d1_noisy = np.asarray([(1 + np.random.normal(0, 1 / 3) * dn_std) * d1[i] for i in range(len_max)])
    synthesized_data: LineNodesData = {"pj": p2_noisy, "qj": q2_noisy, "vj": v2_noisy, "cj": c2_noisy, "vi": v1_noisy,
                                       "di": d1_noisy}
    return synthesized_data
