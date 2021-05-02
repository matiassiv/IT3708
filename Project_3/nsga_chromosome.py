import numpy as np
import time
from heapq import heappush, heappop
from numba import jit
from random import random, randint

from image import load_image
from utils import (
    arr_idx_to_genotype_idx,
    color_distance,
    connectivity,
    edge_value,
    find_avg_color,
    find_centroid_dist,
    find_segments,
    genotype_idx_to_arr_idx,
    get_neighbor_from_dir,
    get_neighbours_from_other_segment,
    get_neighbours_from_same_segment,
    mst, 
    overall_deviation,
    rgb_euclidean
)

class Chromosome:
    def __init__(self, arr, genotype=None):

        self.rows = arr.shape[0]
        self.cols = arr.shape[1]
        # Generate initial phenotype
        if genotype is None:
            self.genotype = mst(arr, self.rows, self.cols)
           
        else:
            self.genotype = genotype
        
        self.segments = find_segments(self.genotype)
        self.num_segments = max(self.segments) + 1
        self.edge_value = -edge_value(arr, self.segments, self.rows, self.cols) # negative to minimize
        self.connectivity = connectivity(arr, self.segments, self.rows, self.cols)
        self.deviation = overall_deviation(arr, self.segments, self.cols)

    # Overload < operator to perform sort on list of Chromosomes
    def __lt__(self, other):
        if self.rank < other.rank:
            return True
        elif self.rank == other.rank and self.distance > other.distance:
            return True
        else:
            return False



#@jit(nopython=True)
def crossover(p1_genotype, p2_genotype, cutoff):
    genotype_length = len(p1_genotype)
    child_genotype = [i for i in range(genotype_length)]
    for i in range(genotype_length):
        if i < cutoff:
            child_genotype[i] = p1_genotype[i]
        else:
            child_genotype[i] = p2_genotype[i]
    
    return child_genotype

#@jit(nopython=True)
def mutate(genotype, num_rows, num_cols, arr, constraints, mut_rate=0.15):
    segments = find_segments(genotype)
    r = random()
    if r < 0.45 and max(segments) >= constraints[0]:
        v_from, v_to = merge_2_similar_segments(arr, segments, num_rows, num_cols)
        seg_1, seg_2 = segments[v_from], segments[v_to]
        
        if seg_1 != seg_2:
            # Set all seg 2 pixels to seg 1
            segment_indices = np.flatnonzero(segments == seg_2)
            segments[segment_indices] = seg_1
            genotype = segment_mst(
                arr, segments, seg_1, np.array(genotype), num_rows, num_cols, 0)
        
    elif r < 0.9 and max(segments) < constraints[1]:
        start = time.time()
        if random() < 0.5:
            segment_id = get_worst_segment(arr, segments, num_rows, num_cols)
        else:
            segment_id = np.random.choice(segments)
        genotype = segment_mst(
            arr, segments, segment_id, np.array(genotype), num_rows, num_cols, 1)
        
    else:
        rand_i = randint(0, len(genotype) - 1)
        mutate_random_edge(rand_i, num_rows, num_cols)

    return genotype

def mutate_random_edge(pixel_to_mutate, num_rows, num_cols):
 
    random_direction = randint(0, 8)
    new_edge = get_neighbor_from_dir(pixel_to_mutate, random_direction, num_rows, num_cols)
    return arr_idx_to_genotype_idx(new_edge, num_cols)

@jit(nopython=True)
def mutate_best_edge(pixel_to_mutate, num_rows, num_cols, arr, segments):
    best = np.inf
    best_pixel = pixel_to_mutate
    a_idx = genotype_idx_to_arr_idx(pixel_to_mutate, num_cols)
    for pixel in get_neighbours_from_other_segment(segments, num_rows, num_cols, pixel_to_mutate):
        neighbor_idx = genotype_idx_to_arr_idx(pixel, num_cols)
        dist = rgb_euclidean(arr, a_idx, neighbor_idx)
        if dist < best:
            best = dist
            best_pixel = pixel
    
    return best_pixel

@jit(nopython=True)
def merge_2_similar_segments(arr, segments, num_rows, num_cols):
    best_edge = (0,0)
    edges = [(0,0) for x in range(0)]
    for i in range(segments.shape[0]):
        for neighbor in get_neighbours_from_other_segment(segments, num_rows, num_cols, i):
            edges.append((i, neighbor))

    if len(edges) > 0:
        centroids = [(0.0,0.0,0.0) for x in range(0)] # Hold all centroid colors
        for i in range(max(segments)+1):
            # Get index of all pixels in segment
            segment_indices = np.flatnonzero(segments == i)
            # Get average color of pixels in segment - tuple of the form (avg_r, avg_g, avg_b)
            avg_color = find_avg_color(arr, segment_indices, num_cols)
            centroids.append(avg_color) # Add centroid color
        
        best_edge = edges[0]
        best_dist = color_distance(centroids[segments[best_edge[0]]], centroids[segments[best_edge[1]]])
        similar_segments = (segments[best_edge[0]], segments[best_edge[1]])
        for j in range(1, len(edges)):
            curr_edge = edges[j]
            curr_dist = color_distance(centroids[segments[curr_edge[0]]], centroids[segments[curr_edge[1]]])
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_edge = curr_edge
                similar_segments = (segments[curr_edge[0]], segments[curr_edge[1]])
    return best_edge

@jit(nopython=True)
def get_worst_segment(arr, segments, num_rows, num_cols):

    # Calculate deviation for each segment
    centroid_color = [(0.0, 0.0, 0.0) for x in range(0)]
    avg_deviation = [0.0 for x in range(0)]
    for i in range(max(segments)+1):
        tot_deviation = 0.0
        # Get index of all pixels in segment
        segment_indices = np.flatnonzero(segments == i)

        # Get average color of pixels in segment - tuple of the form (avg_r, avg_g, avg_b)
        avg_color = find_avg_color(arr, segment_indices, num_cols)
        centroid_color.append(avg_color)
        for pixel_idx in segment_indices:
            arr_idx = genotype_idx_to_arr_idx(pixel_idx, num_cols)
            # Compare pixel with centroid color and add to total deviation
            tot_deviation += find_centroid_dist(arr, arr_idx, avg_color)
        avg_deviation.append(tot_deviation/segment_indices.shape[0])

    return avg_deviation.index(max(avg_deviation))

@jit(nopython=True)
def segment_mst(arr, segments, segment_to_span, genotype, num_rows, num_cols, num_segments=0):
    """
    Generates a minimum spanning tree from a randomized starting location
    Returns mst encoded as a genotype for GA chromosome
    Returns: [int]
    """
    segment_indices = np.flatnonzero(segments == segment_to_span)
    num_pixels = segment_indices.shape[0]
    #genotype = [i for i in range(num_pixels)]
    visited = set()
    # Get random start node
    current = np.random.choice(segment_indices)

    # Force type inference for numba!
    edgeQueue = [(0.0, 0, 0) for x in range(0)]
    worst_edges = [(0.0, 0, 0) for x in range(0)]
    while len(visited) < num_pixels:
        # Add current to visited if not previously added
        if current not in visited:
            visited.add(current)
            # Add edges to priorityqueue
            curr_idx = genotype_idx_to_arr_idx(current, num_cols)
            edges = get_neighbours_from_same_segment(segments, num_rows, num_cols, current)
            for edge in edges:
                edge_idx = genotype_idx_to_arr_idx(edge, num_cols)
                dist = rgb_euclidean(arr, curr_idx, edge_idx)
                heappush(edgeQueue, (dist, current, edge))
        
        # Get edge with minimum cost
        edge_tuple = heappop(edgeQueue)
        cost = edge_tuple[0]
        v_from = edge_tuple[1]
        v_to = edge_tuple[2]

        if v_to not in visited:
            genotype[v_to] = v_from
            worst_edges.append(edge_tuple)
        
        current = v_to

    worst_edges.sort(reverse=True)
    for i in range(num_segments):
        v_to = worst_edges[i][2]
        genotype[v_to] = v_to
    return genotype

def parent_selection(population):
    pop_size = len(population)
    p1_idx = randint(0, pop_size-1)
    p2_idx = randint(0, pop_size-1)

    if population[p1_idx] < population[p2_idx]: # Uses overloaded function from class definition
        return population[p1_idx]
    else:
        return population[p2_idx]

@jit(nopython=True)
def check_constraints(p, constraints):
    if p[4] >= constraints[0] and p[4] <= constraints[1]:
        return True
    return False

@jit(nopython=True)
def check_dominates(p, q, constraints):
    """
    p and q are numpy arrays: [edge_value, connectivity, deviation, pop_index]
    """
    if check_constraints(p, constraints) and not check_constraints(q, constraints):
        return True
    elif p[0] < q[0] and p[1] <= q[1] and p[2] <= q[2]:
        return True
    elif p[0] <= q[0] and p[1] < q[1] and p[2] <= q[2]:
        return True
    elif p[0] <= q[0] and p[1] <= q[1] and p[2] < q[2]:
        return True
    else:
        return False

@jit(nopython=True)
def nd_sort(pop, constraints):
    """
    pop has the shape (num_chromosomes, 5)
    where each row is a numpy array with the objectives and population index
    [-edge_value, connectivity, deviation, pop_index, num_segments]
    constraints is a tuple of the form (min_segments, max_segments)
    Returns:
    F - dictionary of frontiers with indices of population for each frontier
    rank - list where index of list corresponds with index in pop, and value is rank of individual
    """
    # Dictionary of solutions with key p, where value is list of solutions
    # that are dominated by p
    S = {}                 
    n = [0 for i in range(pop.shape[0])] # Domination counter for an individual p
    F = {} # Dictionary for frontiers containing lists of indices in a frontier
    ranks = [0 for i in range(pop.shape[0])] # ranks tracker - essentially inverse of F dict

    F_1 = [0 for x in range(0)] # Initialize dict and list for numba
    # Iterate over 
    for p in range(pop.shape[0]):
        # Get index and initialise S[p] to empty list
        #p_idx = int(pop[p,3])
        S_p = [0 for x in range(0)] # Initialise int list for numba

        for q in range(pop.shape[0]):
            if check_dominates(pop[p], pop[q], constraints):
                S_p.append(q)
            elif check_dominates(pop[q], pop[p], constraints):
                n[p] += 1

        S[p] = np.array(S_p)

        if n[p] == 0:
            ranks[p] = 1
            F_1.append(p)
    F[1] = np.array(F_1) # Numba currently doesn't support list as value in dict
    i = 1
    while len(F[i]) > 0:
        Q = [0 for x in range(0)]
        for p in F[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    ranks[q] = i + 1
                    Q.append(q)
        i += 1
        F[i] = np.array(Q)
    
    return F, ranks

@jit(nopython=True)
def crowding_distance(frontier):
    """
    solutions is a 2d numpy array with shape (num_frontier_solutions, 4)
    where cols are [-edge_val, connectivity, deviation, pop_idx, num_segments]
    """
    l = frontier.shape[0] # Get number of solutions in frontier
    distances = [0.0 for x in range(l)]

    
    for m in range(3):
        sorted_idx = np.argsort(frontier[:, m])
        
        frontier_m = frontier[sorted_idx]
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[l-1]] = np.inf
        for i in range(1, l-1):
            f_idx = sorted_idx[i]
            diff = frontier_m[i+1,m] - frontier_m[i-1,m]

            range_approx = frontier_m[l-1,m] - frontier_m[0,m] # Hard to find objective min and max
            distances[f_idx] = distances[f_idx] + diff /(range_approx + 1e-4) # add small eps to avoid 0 division
    
    return distances



#def crowding_comparison(c1, c2):
    