# To get results from the paper run the following command:
#  python real_data_joint_estimation.py --dataset_folder reichstag_data --skip_pairs 1 --num_inlier_kps 60 --outlier_rate 0.4 --figure_path reichstag_inlier_estimation_hist.pdf

from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from colmap.scripts.python.read_write_dense import read_array
from imageio import imread
from tqdm import tqdm

import h5py
import deepdish as dd
from time import time

import cv2 as cv
from scipy.optimize import linear_sum_assignment
from ortools.graph import pywrapgraph

import argparse
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import json

from ortools.graph import pywrapgraph

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', required=True)
parser.add_argument('--skip_pairs', default=1, type=int)
parser.add_argument('--skip_kps', default=1, type=int)
parser.add_argument('--outlier_rate', default=0.0, type=float)
parser.add_argument('--num_inlier_kps', default=100.0, type=int)
parser.add_argument('--figure_path', required=True)
args = parser.parse_args()

def dist(x, y):
    d =  np.linalg.norm(x - y) ** 2
    return d if d > 0 else 1e-10

def num_correct_matches(pie, pie_star, num_inliers):
    return np.sum(pie[:num_inliers] == pie_star[:num_inliers])

def num_in_out_matches(pie, pie_star, num_inliers):
    return np.sum(pie[:num_inliers] >= num_inliers)

def num_out_out_matches(pie, pie_star, num_inliers):
    return np.sum(pie[num_inliers:] >= num_inliers)

def num_out_in_matches(pie, pie_star, num_inliers):
    return np.sum((pie[num_inliers:] < num_inliers) & (pie[num_inliers:] >= 0))

def num_inliers_matched(pie, pie_star, num_inliers):
    return np.sum(pie[:num_inliers] >= 0)

def num_outliers_matched(pie, pie_star, num_inliers):
    return np.sum(pie[num_inliers:] >= 0)

def greedy(X, X_hash, num_to_match=None, return_cost=False, progress_bar=False):
    costs = [0]
    cost = 0
    if num_to_match is None:
        num_to_match = len(X)
    n, m = len(X), len(X_hash)
    M = np.zeros((n, m))
    if progress_bar:
        iterable = tqdm(enumerate(X), desc='Calculating pairwise distances', total=n)
    else:
        iterable = enumerate(X)
    for i, x in iterable:
        for j, x_hash in enumerate(X_hash):
            M[i, j] = dist(x, x_hash)
    pie = -1 * np.ones(n, dtype=np.int32)
    for _ in range(num_to_match):
        i, j = np.unravel_index(M.argmin(), M.shape)
        pie[i] = j
        cost += M[i, j]
        costs.append(cost)
        M[:, j] = np.inf
        M[i, :] = np.inf

    if return_cost:
        return pie, costs

    return pie

def MinCostFlow(X, X_hash, num_to_match = None, scale = 1000, sigma=None, return_cost=False):
    n, m = (len(X), len(X_hash))
    M = np.zeros(n * m)
    for i, x in enumerate(X):
        for j, x_hash in enumerate(X_hash):
            M[i * m + j] = dist(x, x_hash)
        
    if sigma is not None:
        M = M / 2 / sigma / sigma
    
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    
    start_nodes = np.concatenate((np.zeros(n), 
                                  np.repeat(np.arange(1, n + 1), m), 
                                  np.arange(n + 1, n + m + 1))).astype(int).tolist()
    
    end_nodes = np.concatenate((np.arange(1, n + 1), 
                                np.tile(np.arange(n + 1, n + m + 1), n), 
                                np.ones(m) * (n + m + 1))).astype(int).tolist()
    
    num_overall_edges = (n + 1) * (m + 1) - 1
    capacities = np.ones(num_overall_edges).astype(int).tolist()
    costs = (np.concatenate((np.zeros(n), M * scale ,np.zeros(m))).astype(int).tolist())
    
    source = 0
    sink = n + m + 1
    if num_to_match is None:
        num_to_match = n
    
    supplies = [num_to_match] + [0] * (n + m) + [-num_to_match]
    
    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i],
                                                    end_nodes[i], capacities[i],
                                                    costs[i])
    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])
    
    status = min_cost_flow.Solve()
    assignment = -1 * np.ones(n, dtype=np.int32)
    if status == min_cost_flow.OPTIMAL:
        for arc in range(min_cost_flow.NumArcs()):
            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc) != source and min_cost_flow.Head(arc) != sink:
                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.
                if min_cost_flow.Flow(arc) > 0:
                    assignment[min_cost_flow.Tail(arc) - 1] = min_cost_flow.Head(arc) - n - 1
        
    else:
        print('There was an issue with the min cost flow input.')
        print(f'Status: {status}')

    if return_cost:
        return assignment, min_cost_flow.OptimalCost() / scale
    else:
        return assignment


def LSS(X, X_hash):
    M = np.zeros((len(X), len(X_hash)))
    for i, x in enumerate(X):
        for j, x_hash in enumerate(X_hash):
            M[i, j] = dist(x, x_hash)
            
    row_ind, col_ind = linear_sum_assignment(M)
    return col_ind

def LSNS(X, X_hash, sigma, sigma_hash):
    M = np.zeros((len(X), len(X_hash)))
    for i, x in enumerate(X):
        for j, x_hash in enumerate(X_hash):
            M[i, j] = dist(x, x_hash) / (sigma_hash[j] ** 2 + sigma[i] ** 2) ** (1/2)
    
    row_ind, col_ind = linear_sum_assignment(M)
    return col_ind

def LSL(X, X_hash):
    M = np.zeros((len(X), len(X_hash)))
    for i, x in enumerate(X):
        for j, x_hash in enumerate(X_hash):
            M[i, j] = np.log(dist(x, x_hash))
    
    row_ind, col_ind = linear_sum_assignment(M)
    return col_ind

def estimate_sigma_sq(X, X_hash, k, d, gamma = 3, matcher=greedy, costs=None):
    if costs is not None:
        cost = costs[k] 
    else:
        _, cost = matcher(X, X_hash, k, return_cost=True)
    return cost / (d * k - gamma * np.sqrt(d * k)) # / 2

def threshold(n, d, alpha=0.01):
    return  1/4 * max(d * np.log(4 * n * n / alpha) ** (1/2),
                   np.log(8 * n * n / alpha) )

def k_sig_joint_est(X, X_hash, matcher=greedy, kmin=1):
    n = len(X)
    d = len(X[0])
    thr = threshold(n, d)
    res = {}
    
    _, costs = matcher(X, X_hash, num_to_match=n, return_cost=True)
    
    cost_prev = costs[kmin]
    for k_alpha in range(kmin + 1, n):
        sigma_hat_sq = estimate_sigma_sq(X, X_hash, k_alpha, d, gamma=0.05, matcher=greedy, costs=costs)
        cost = costs[k_alpha]
        if cost - cost_prev > (d + thr) * sigma_hat_sq:
            break
        cost_prev = cost
            
    res['k_alpha'] = k_alpha
    res['sigma_hat'] = np.sqrt(sigma_hat_sq / 2)
    return res

src = args.dataset_folder

# load reconstruction from colmap
cameras, images, points = read_model(path=src + '/dense/sparse', ext='.bin')

print(f'Cameras: {len(cameras)}')
print(f'Images: {len(images)}')
print(f'3D points: {len(points)}')

indices = [i for i in cameras]

# Retrieve one image, the depth map, and 2D points
def get_image(idx, verbose=False):
    im = imread(src + '/dense/images/' + images[idx].name)
    depth = read_array(src + '/dense/stereo/depth_maps/' + images[idx].name + '.photometric.bin')
    min_depth, max_depth = np.percentile(depth, [5, 95])
    depth[depth < min_depth] = min_depth
    depth[depth > max_depth] = max_depth

    # reformat data
    q = images[idx].qvec
    R = qvec2rotmat(q)
    T = images[idx].tvec
    p = images[idx].xys
    pars = cameras[idx].params
    K = np.array([[pars[0], 0, pars[2]], [0, pars[1], pars[3]], [0, 0, 1]])
    pids = images[idx].point3D_ids
    v = pids >= 0
    if verbose:
        print('Number of (valid) points: {}'.format((pids > -1).sum()))
        print('Number of (total) points: {}'.format(v.size))
    
    # get also the clean depth maps
    base = '.'.join(images[idx].name.split('.')[:-1])
    with h5py.File(src + '/dense/stereo/depth_maps_clean_300_th_0.10/' + base + '.h5', 'r') as f:
        depth_clean = f['depth'][()] #.value

    return {
        'image': im,
        'depth_raw': depth,
        'depth': depth_clean,
        'K': K,
        'q': q,
        'R': R,
        'T': T,
        'xys': p,
        'ids': pids,
        'valid': v}


# We can just retrieve all the 3D points
xyz, rgb = [], []
for i in points:
    xyz.append(points[i].xyz)
    rgb.append(points[i].rgb)
xyz = np.array(xyz)
rgb = np.array(rgb)

print(xyz.shape)

# We also provide a measure of how images overlap, based on the bounding boxes
# of the 2D points they have in common
t = time()
# each pair contains [bbox1, bbox2, visibility1, visibility2, # of shared matches]
pairs = dd.io.load(src + '/dense/stereo/pairs-dilation-0.00-fixed2.h5')
print(f'Done ({time() - t:.2f} s.)')

# Threshold at a given value
# pairs[p][0]: ratio between the area of the bounding box containing common points and that of image 1
# pairs[p][1]: same for image 2
th = 0.3

filtered = []
for p in pairs:
    if pairs[p][0] >= th and pairs[p][1] >= th:
        idx1, idx2 = p
        filtered += [p]
print(f'Valid pairs: {len(filtered)}/{len(pairs)}')
pairs = filtered

method_names = ['Greedy', 'LSS', 'Greedy_Correct_K', 'MinCostFlow_Correct_K']
error_names = ['correct', 'in_out', 'out_in', 'out_out','in_matched', 'out_matched']
error_fns = [num_correct_matches, num_in_out_matches, num_out_in_matches,
             num_out_out_matches, num_inliers_matched, num_outliers_matched]

res = {'num_in': {},
       'num_out': {}}

res.update({f'{method}_{error}': {} for method in method_names for error in error_names})

estimates = []

random.shuffle(pairs)
for idx1, idx2 in tqdm(pairs[::args.skip_pairs]):
    
    # pick one pair (e.g. the third one)
    # These two images should be matchable
    data1 = get_image(idx1)
    data2 = get_image(idx2)

    # Find the points in common
    v1 = data1['ids'][data1['ids'] > 0]
    v2 = data2['ids'][data2['ids'] > 0]
    common = np.intersect1d(v1, v2)
    
    depth1 = data1['depth']
    K1 = data1['K']
    R1 = data1['R']
    T1 = data1['T']

    depth2 = data2['depth']
    K2 = data2['K']
    R2 = data2['R']
    T2 = data2['T']

    # Get the points from one of the images
    xy1s = np.array([tmp_xy for i, tmp_xy in enumerate(data1['xys']) if data1['ids'][i] in common]) #data1['xys'][data1['valid'], :]
    u_xy1s = xy1s.T
    
    # Filter wrong xys
    u_xy1s = u_xy1s[:, u_xy1s[0] >= 0]
    u_xy1s = u_xy1s[:, u_xy1s[0] < data1['depth'].shape[1] ]
    u_xy1s = u_xy1s[:, u_xy1s[1] >= 0]
    u_xy1s = u_xy1s[:, u_xy1s[1] < data1['depth'].shape[0] ]

    # Convert to homogeneous coordinates
    u_xy1s = np.concatenate([u_xy1s, np.ones([1, u_xy1s.shape[1]])], axis=0)

    # Get depth (on image 1) for each point
    u_xy1s_int = u_xy1s.astype(np.int32)
    z1 = data1['depth'][u_xy1s_int[1], u_xy1s_int[0]]

    # Eliminate points on occluded areas
    not_void = z1 > 0
    u_xy1s = u_xy1s[:, not_void]
    z1 = z1[not_void]

    # Move to world coordinates
    n_xyz1s = np.dot(np.linalg.inv(K1), u_xy1s)
    n_xyz1s = n_xyz1s * z1 / n_xyz1s[2, :]
    xyz_w = np.dot(R1.T, n_xyz1s - T1[:,None])

    # Reproject into image 2
    n_xyz2s = np.dot(R2, xyz_w) + T2[:,None]
    u_xy2s = np.dot(K2, n_xyz2s)
    z2 = u_xy2s[2,:]
    u_xy2s = u_xy2s / z2

    # Get SIFT descriptors
    query_kps, train_kps = ([cv.KeyPoint(x = xys[0], y = xys[1], size=10) for xys in u_xy1s[[0, 1], ::args.skip_kps].T],
                            [cv.KeyPoint(x = xys[0], y = xys[1], size=10) for xys in u_xy2s[[0, 1], ::args.skip_kps].T])
    
    if len(query_kps) != len(train_kps):
        print("Something is wrong with kps in pair {} - {}".format(idx1, idx2))
        exit(0)
    
    initial_number_kps = len(query_kps)
    num_inlier_kps = args.num_inlier_kps
    num_outlier_kps = int(num_inlier_kps * args.outlier_rate / (1 - args.outlier_rate))
    
    if initial_number_kps / 3 < num_inlier_kps + num_outlier_kps * 2:
        print('Skipping pair {} - {} because of lack of query kps'.format(idx1, idx2))
        continue

    # Filtering given number of inlier keypoints
    initial_rand_permutation = np.random.permutation(initial_number_kps)
    inlier_kp_ids = initial_rand_permutation[:num_inlier_kps]
    train_outlier_kp_ids = initial_rand_permutation[num_inlier_kps: num_inlier_kps + num_outlier_kps]
    query_outlier_kp_ids = initial_rand_permutation[-num_outlier_kps: ]
    
    if len(inlier_kp_ids) != num_inlier_kps or len(train_outlier_kp_ids) != num_outlier_kps or len(query_outlier_kp_ids) != num_outlier_kps:
        print("Something is wrong with kp ids in pair {} - {}".format(idx1, idx2))
        print(len(inlier_kp_ids), num_inlier_kps, len(train_outlier_kp_ids), num_outlier_kps, len(query_outlier_kp_ids))
        
    
    train_inlier_kps = [train_kps[i] for i in inlier_kp_ids]
    train_outlier_kps = [train_kps[i] for i in train_outlier_kp_ids]
    query_inlier_kps = [query_kps[i] for i in inlier_kp_ids]
    query_outlier_kps = [query_kps[i] for i in query_outlier_kp_ids]
    train_kps = train_inlier_kps + train_outlier_kps
    query_kps = query_inlier_kps + query_outlier_kps
    
    correct_matching = list(range(num_inlier_kps))


    sift = cv.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.compute(data1['image'], query_kps)
    kp2, desc2 = sift.compute(data2['image'], train_kps)
    
    estimates.append(k_sig_joint_est(desc1, desc2, kmin=2))

plt.hist([t['k_alpha'] for t in estimates], bins=range(0, 100)[::3], density=True);
plt.xticks(fontsize=20)
plt.xlabel(r'$\widehat{k}  $', fontsize=20);
ticks, labels = plt.yticks()
plt.yticks(ticks[::2], fontsize=15);
plt.gcf().subplots_adjust(bottom=0.2)
plt.subplots_adjust(bottom=0.2)
plt.savefig(args.figure_path, dpi=10, format='pdf')
    