import numpy as np

from heapq import heappush, heappop
from image import load_image
from numba import jit
from math import sqrt
from random import randint
import time

"""
Util functions mostly decorated by numba and used 
in calculating segments and fitness score components
"""

@jit(nopython=True)
def arr_idx_to_genotype_idx(arr_idx, num_cols):
    """
    Converts array indices from pixel array to
    corresponding index in chromosome
    Returns: int
    """
    return arr_idx[0] * num_cols + arr_idx[1]

@jit(nopython=True)
def genotype_idx_to_arr_idx(geno_idx, num_cols):
    """
    Converts index in chromosome to corresponding
    index in pixel array.
    Returns: (row_index, col_index)
    """
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
    """
    Gets all edges from a given chromosome index. Excludes edge to itself.
    Returns list of all pixels reachable from itself.
    Returns: [(r1, c1), ..., (rn, cn)]
    """
    edges = [
        get_neighbor_from_dir(geno_idx, d, num_rows, num_cols) for d in range(9)]
    original = genotype_idx_to_arr_idx(geno_idx, num_cols)
    edges = [e for e in edges if e != original]
    return edges

@jit(nopython=True)
def rgb_euclidean(arr, p_1, p_2):
    """
    Calculates euclidean distance between pixel colors of two pixels
    Returns: Float
    """
    r1, c1 = p_1[0], p_1[1]
    r2, c2 = p_2[0], p_2[1]
    d0 = arr[r1, c1, 0] - arr[r2, c2, 0]
    d1 = arr[r1, c1, 1] - arr[r2, c2, 1]
    d2 = arr[r1, c1, 2] - arr[r2, c2, 2]

    return sqrt(d0**2 + d1**2 + d2**2)

@jit(nopython=True)
def mst(arr, num_rows, num_cols, num_segments=0):
    """
    Generates a minimum spanning tree from a randomized starting location
    Returns mst encoded as a genotype for GA chromosome
    Returns: [int]
    """
    num_pixels = num_rows * num_cols
    genotype = [i for i in range(num_pixels)]
    visited = set()
    # Get random start node
    current = randint(0, num_pixels-1)

    # Force type inference for numba!
    edgeQueue = [(0.0, 0, 0) for x in range(0)]
    worst_edges = [(0.0, 0, 0) for x in range(0)]
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

# @jit(nopython=True) jit is bad cuz of reflected list when passing in chromosome
# function is still quite fast tho, so might not be necessary to begin with
def find_segments(chromosome):
    """
    Finds segments for the mst encoded as a chromosome. 
    Segments-array is a list of ints, where max int value == num segments
    Index of segment array is the corresponding index in chromosome, so
    segment[i] corresponds to the segment pixel i belongs to.
    Returns: np.array([int]) 
    Numpy array is needed for subsequent functions with numba decorators
    """
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
    
    return np.array(segments)

@jit(nopython=True)
def get_neighbours_from_other_segment(segments, num_rows, num_cols, origin):
    """
    Gets the neighbors of a pixel, if the neighbor belongs to a different segment
    than the pixel.
    Returns: [int] where list can be empty
    """
    neighbors = [
        arr_idx_to_genotype_idx(get_neighbor_from_dir(origin, d, num_rows, num_cols), num_cols) 
        for d in range(1, 9)
    ]
    return [x for x in neighbors if segments[x] != segments[origin]]

@jit(nopython=True)
def get_neighbours_from_same_segment(segments, num_rows, num_cols, origin):
    """
    Gets the neighbors of a pixel, if the neighbor belongs to a different segment
    than the pixel.
    Returns: [int] where list can be empty
    """
    neighbors = [
        arr_idx_to_genotype_idx(get_neighbor_from_dir(origin, d, num_rows, num_cols), num_cols) 
        for d in range(1, 9)
    ]
    return [x for x in neighbors if segments[x] == segments[origin] and x != origin]

@jit(nopython=True)
def edge_value(pixel_arr, segments, num_rows, num_cols):
    """
    Calculates total edge value for the entire image, which is the
    total euclidean distance between all neighboring pixels belonging 
    to different segments:
    Returns: Float
    """
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
    """
    Calculates total connectivity of the image, i.e. the degree that
    neighboring pixels are placed in the same segment
    Returns: Float
    """
    tot_connectivity = 0
    for i in range(len(segments)):
        # Get all neighbors of a pixel if neighbor is of a different segment
        neighbors = get_neighbours_from_other_segment(segments, num_rows, num_cols, i)
        tot_connectivity += 0.125 * len(neighbors)
    
    return tot_connectivity

@jit(nopython=True)
def find_avg_color(pixel_arr, segment_indices, num_cols):
    """
    Calculates the "centroid color", i.e. the average color of an
    entire segment for each of the 3 color channels
    Returns: (Float, Float, Float)
    """
    tot_pixels = segment_indices.shape[0]
    tot_red, tot_green, tot_blue = 0, 0, 0
    for pixel in segment_indices:
        arr_idx = genotype_idx_to_arr_idx(pixel, num_cols)
        tot_red += pixel_arr[arr_idx[0], arr_idx[1], 0]
        tot_green += pixel_arr[arr_idx[0], arr_idx[1], 1]
        tot_blue += pixel_arr[arr_idx[0], arr_idx[1], 2]
    
    return tot_red/tot_pixels, tot_green/tot_pixels, tot_blue/tot_pixels

@jit(nopython=True)
def find_centroid_dist(arr, pixel, centroid_color):
    """
    Calculates the euclidean rgb distance between a pixel
    and the centroid color of the segment it belongs to.
    Returns: Float
    """
    r1, c1 = pixel[0], pixel[1]
    d0 = arr[r1, c1, 0] - centroid_color[0]
    d1 = arr[r1, c1, 1] - centroid_color[1]
    d2 = arr[r1, c1, 2] - centroid_color[2]

    return sqrt(d0**2 + d1**2 + d2**2)

@jit(nopython=True)
def color_distance(color_1, color_2):
    d0 = color_1[0] - color_2[0]
    d1 = color_1[1] - color_2[1]
    d2 = color_1[2] - color_2[2]

    return sqrt(d0**2 + d1**2 + d2**2)

@jit(nopython=True)
def overall_deviation(pixel_arr, segments, num_cols):
    """
    Calculates overall deviation of pixel colors in the image segments.
    Pixels are compared to the centroid color of the segment they belong to, and
    added to the total deviation. After all segments have been tallied, the total
    deviation is returned.
    Returns: Float
    """
    tot_deviation = 0.0
    for i in range(max(segments)+1):
        # Get index of all pixels in segment
        segment_indices = np.flatnonzero(segments == i)

        # Get average color of pixels in segment - tuple of the form (avg_r, avg_g, avg_b)
        avg_color = find_avg_color(pixel_arr, segment_indices, num_cols)
        for pixel_idx in segment_indices:
            arr_idx = genotype_idx_to_arr_idx(pixel_idx, num_cols)
            # Compare pixel with centroid color and add to total deviation
            tot_deviation += find_centroid_dist(pixel_arr, arr_idx, avg_color)
    
    return tot_deviation

@jit(nopython=True)
def isEdge(segments, num_rows, num_cols, pixel):
    """
    Returns true if pixel is an edge, i.e. neighbor's a different segment
    than itself.
    Returns: Boolean
    """
    pixel = arr_idx_to_genotype_idx(pixel, num_cols)
    n = get_neighbours_from_other_segment(segments, num_rows, num_cols, pixel)
    if len(n) > 0:
        return True
    return False

def main():
    arr = load_image("training_images/86016/Test image.jpg")
    print(arr.shape)

    start_time = time.time()
    for i in range(100):
        c = mst(arr, arr.shape[0], arr.shape[1])
        c[12000] = 12000
        c[10000] = 10000
        c[4500] = 4500
        s = np.array(find_segments(c))
        e = edge_value(arr, s, arr.shape[0], arr.shape[1])
        c = connectivity(arr, s, arr.shape[0], arr.shape[1])
        d = overall_deviation(arr, s, arr.shape[1])
    print("Time elapsed:", time.time() - start_time)
    print(e)
    print(c)
    print(d)

if __name__ == "__main__":
    main()