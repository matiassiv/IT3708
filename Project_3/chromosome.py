import numpy as np
from random import randint
#from utils import rgb_euclidean
from heapq import heappush, heappop
from image import load_image
from numba import jit
from math import sqrt
import time

class Dirs:
    P  = 0
    E  = 1
    W  = 2
    N  = 3
    S  = 4
    NE = 5
    SE = 6
    NW = 7
    SW = 8

class Chromosome:
    def __init__(self, arr, genotype=None):

        self.rows = arr.shape[0]
        self.cols = arr.shape[1]
        # Generate initial phenotype
        if genotype is None:
            self.genotype = mst(arr, self.rows, self.cols)
        

@jit(nopython=True)
def arr_idx_to_genotype_idx(arr_idx, num_cols):
    return arr_idx[0] * num_cols + arr_idx[1]

@jit(nopython=True)
def genotype_idx_to_arr_idx(geno_idx, num_cols):
    # Returns a tuple (row_index, col_index) of a pixel in genotype 
    return (geno_idx // num_cols, geno_idx % num_cols)

@jit(nopython=True)
def get_neighbor_from_dir(geno_idx, d, num_rows, num_cols):
    # Gets index of neighbor pixel based on direction d
    # Takes in index of genotype 
    # Returns (row, col) of neighbor in direction d
    # If no neighbor, (r, c) of itself is returned 
    if d == 1:
        arr_idx = genotype_idx_to_arr_idx(geno_idx, num_cols)
        row = arr_idx[0]
        col = arr_idx[1] + 1
        if col < num_cols:
            # Return neighbor
            return (row, col)
        
    elif d == 2:
        arr_idx = genotype_idx_to_arr_idx(geno_idx, num_cols)
        row = arr_idx[0]
        col = arr_idx[1] - 1
        if col >= 0:
            # Return neighbor
            return (row, col)
        
    elif d == 3:
        arr_idx = genotype_idx_to_arr_idx(geno_idx, num_cols)
        row = arr_idx[0] - 1
        col = arr_idx[1]
        if row >= 0:
            # Return neighbor
            return (row, col)

    elif d == 4:
        arr_idx = genotype_idx_to_arr_idx(geno_idx, num_cols)
        row = arr_idx[0] + 1
        col = arr_idx[1]
        if row < num_rows:
            # Return neighbor
            return (row, col)
        
    elif d == 5:
        arr_idx = genotype_idx_to_arr_idx(geno_idx, num_cols)
        row = arr_idx[0] - 1
        col = arr_idx[1] + 1
        if row >= 0 and col < num_cols:
            # Return neighbor
            return (row, col)
        
    elif d == 6:
        arr_idx = genotype_idx_to_arr_idx(geno_idx, num_cols)
        row = arr_idx[0] + 1
        col = arr_idx[1] + 1
        if row < num_rows and col < num_cols:
            # Return neighbor
            return (row, col)
        
    elif d == 7:
        arr_idx = genotype_idx_to_arr_idx(geno_idx, num_cols)
        row = arr_idx[0] - 1
        col = arr_idx[1] - 1
        if row >= 0 and col >= 0:
            # Return neighbor
            return (row, col)

    elif d == 8:
        arr_idx = genotype_idx_to_arr_idx(geno_idx, num_cols)
        row = arr_idx[0] + 1
        col = arr_idx[1] - 1
        if row < num_rows and col >= 0:
            # Return neighbor
            return (row, col)

    
    return genotype_idx_to_arr_idx(geno_idx, num_cols)

@jit(nopython=True)
def getEdges(geno_idx, num_rows, num_cols):
    edges = [
        get_neighbor_from_dir(geno_idx, d, num_rows, num_cols) for d in range(9)]
    original = genotype_idx_to_arr_idx(geno_idx, num_cols)
    edges = [e for e in edges if e != original]
    return edges

@jit(nopython=True)
def rgb_euclidean(arr, p_1, p_2):
    r1, c1 = p_1[0], p_1[1]
    r2, c2 = p_2[0], p_2[1]
    d0 = arr[r1, c1, 0] - arr[r2, c2, 0]
    d1 = arr[r1, c1, 1] - arr[r2, c2, 1]
    d2 = arr[r1, c1, 2] - arr[r2, c2, 2]

    return sqrt(d0**2 + d1**2 + d2**2)

@jit(nopython=True)
def mst(arr, num_rows, num_cols):
    num_pixels = num_rows * num_cols
    genotype = [i for i in range(num_pixels)]
    visited = set()
    # Get random start node
    current = randint(0, num_pixels-1)

    # Force type inference for numba!
    edgeQueue = [(0.0, 0, 0) for x in range(0)]

    while len(visited) < num_pixels:
        # Add current to visited if not previously added
        if current not in visited:
            visited.add(current)
            # Add edges to priorityqueue
            edges = getEdges(current, num_rows, num_cols)
            for edge in edges:
                start = genotype_idx_to_arr_idx(current, num_cols)
                dist = rgb_euclidean(arr, start, edge)
                end = arr_idx_to_genotype_idx(edge, num_cols)
                heappush(edgeQueue, (dist, current, end))
        
        # Get edge with minimum cost
        edge_tuple = heappop(edgeQueue)
        v_from = edge_tuple[1]
        v_to = edge_tuple[2]

        if v_to not in visited:
            genotype[v_to] = v_from
        
        current = v_to
    
    return genotype

# @jit(nopython=True) jit is bad cuz of reflected list when passing in chromosome
# function is still quite fast tho, so might not be necessary to begin with
def find_segments(chromosome):
    # Initialize segment-id array
    segments = [-1 for i in range(len(chromosome))]
    # Set initial segment-id
    curr_segment_id = 0
    for i in range(len(chromosome)):
        # If already added to segment, then continue
        if segments[i] != -1:
            continue
        
        # Start tracking which pixels are in current segment
        curr_segment = [i]
        segments[i] = curr_segment_id
        next_pixel = chromosome[i]

        # Follow edges until we find a previously segmented pixel
        while segments[next_pixel] == -1:
            curr_segment.append(next_pixel)
            segments[next_pixel] = curr_segment_id
            next_pixel = chromosome[next_pixel]
        
        # If encountered segment is different from current segment, we merge them together
        if (segments[i] != segments[next_pixel]):
            parent_segment = segments[next_pixel]
            for pixel in curr_segment:
                segments[pixel] = parent_segment

        # If encountered is same segment, then we increment and start building another segment
        else:
            curr_segment_id += 1
    
    return segments

@jit(nopython=True)
def get_neighbours_from_other_segment(segments, num_rows, num_cols, origin):
    neighbors = [
        arr_idx_to_genotype_idx(get_neighbor_from_dir(origin, d, num_rows, num_cols), num_cols) 
        for d in range(1, 9)
    ]
    return [x for x in neighbors if segments[x] != segments[origin]]


@jit(nopython=True)
def edge_value(pixel_arr, segments, num_rows, num_cols):
    tot_dist = 0
    for i in range(len(segments)):
        # Get all neighbors of a pixel if neighbor is of a different segment
        neighbors = get_neighbours_from_other_segment(segments, num_rows, num_cols, i)
        pixel_i = genotype_idx_to_arr_idx(i, num_cols)
        for n in neighbors:
            pixel_j = genotype_idx_to_arr_idx(n, num_cols)
            tot_dist += rgb_euclidean(pixel_arr, pixel_i, pixel_j)
    
    return tot_dist

@jit(nopython=True)
def connectivity(pixel_arr, segments, num_rows, num_cols):
    tot_connectivity = 0
    for i in range(len(segments)):
        # Get all neighbors of a pixel if neighbor is of a different segment
        neighbors = get_neighbours_from_other_segment(segments, num_rows, num_cols, i)
        tot_connectivity += 0.125 * len(neighbors)
    
    return tot_connectivity

def main():
    arr = load_image("training_images/86016/Test image.jpg")
    print(arr.shape)
    c = Chromosome(arr)
    c.genotype[12000] = 12000
    #c.genotype[10000] = 10000
    s = np.array(find_segments(c.genotype))
    start_time = time.time()
    for i in range(50):
        e = edge_value(arr, s, arr.shape[0], arr.shape[1])
        c = connectivity(arr, s, arr.shape[0], arr.shape[1])
    print("Time elapsed:", time.time() - start_time)
    print(e)
    print(c)

main()