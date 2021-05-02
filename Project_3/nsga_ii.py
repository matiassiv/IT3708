import numpy as np
import time
from random import random, randint

from nsga_chromosome import (
    Chromosome, 
    crossover,
    crowding_distance,
    mutate,
    nd_sort,
    parent_selection,
)
from utils import isEdge
from image import load_image, save_black_white, save_segmented

def nsga_ii(pixel_arr, constraints, pop_size=70, g_max=45, p_c=0.6, p_m=0.4, verbose=True):

    start_time = time.time()
    num_rows, num_cols = pixel_arr.shape[0], pixel_arr.shape[1]
    population = [Chromosome(pixel_arr) for i in range(pop_size)]
    obj_vect = np.array([
        [c.edge_value, c.connectivity, c.deviation, i, c.num_segments] for i, c in enumerate(population)
        ], dtype=np.float64)
    
    # Get frontiers and ranks of population through non-dominated sort
    frontiers, ranks = nd_sort(obj_vect, constraints)
    for i in range(pop_size):
        population[i].rank = ranks[i]

    # Get crowding distances and assign to respective chromosome
    for i in range(1, len(frontiers) + 1):
        idx = frontiers[i]
        if len(idx) > 0:
            distances = crowding_distance(obj_vect[frontiers[i]])
        for j in range(len(distances)):
            pop_idx = int(obj_vect[j,3])
            population[pop_idx].distance = distances[j]
    population.sort()
    best_ind = population[0]

    if verbose:
        print("Generation 0:")
        print("Number of segmentations:", best_ind.num_segments)
        print("Edge value:", -best_ind.edge_value)
        print("Connectivity:", best_ind.connectivity)
        print("Deviation:", best_ind.deviation)
        print(f"Initial generation time: {time.time() - start_time}\n", flush=True)
    
    for g in range(1, g_max+1):
        start_time = time.time()
        offspring = []
        for i in range(pop_size//2):
            p1 = parent_selection(population)
            p2 = parent_selection(population)
            if random() < p_c:
                cutoff = randint(0, len(p1.genotype)-1)
                c1_geno = crossover(p1.genotype, p2.genotype, cutoff)
                c2_geno = crossover(p2.genotype, p1.genotype, cutoff)
            else:
                c1_geno = p1.genotype
                c2_geno = p2.genotype
            
            if random() < p_m:
                c1_geno = mutate(c1_geno, num_rows, num_cols, arr, constraints)
            if random() < p_m:
                c2_geno = mutate(c2_geno, num_rows, num_cols, arr, constraints)
           
            c1 = Chromosome(pixel_arr, c1_geno)
            c2 = Chromosome(pixel_arr, c2_geno)
        
            offspring.append(c1)
            offspring.append(c2)   

        # Combine offspring and parent population
        population = population + offspring
        
        # Gather necessary stats for individuals in np array for numba functions
        obj_vect = np.array([
            [c.edge_value, c.connectivity, c.deviation, i, c.num_segments] for i, c in enumerate(population)
            ], dtype=np.float64)
      
        # Get frontiers of population by non-dominated sort
        frontiers, ranks = nd_sort(obj_vect, constraints)
        for i in range(len(population)):
            population[i].rank = ranks[i]
       
        # Get crowding distances and assign to respective chromosome
        for i in range(1, len(frontiers) + 1):
            idx = frontiers[i]
            if len(idx) > 0:
                distances = crowding_distance(obj_vect[frontiers[i]])
                for j in range(len(distances)):
                    pop_idx = int(obj_vect[frontiers[i]][j,3])
                    population[pop_idx].distance = distances[j]
        
        # Sort by rank and distance and perform simple elitist replacement
        population.sort()
        population = population[:pop_size]
        best_ind = population[0]

        if verbose:
            print(f"Generation {g}:")
            print("Number of segmentations:", best_ind.num_segments)
            print("Edge value:", -best_ind.edge_value)
            print("Connectivity:", best_ind.connectivity)
            print("Deviation:", best_ind.deviation)
            print("No rank 1s:", sum([c.rank if c.rank==1 else 0 for c in population ]))
            print(f"Time elapsed: {time.time() - start_time}\n", flush=True)
    
    return population
    
if __name__ == "__main__":
    path = "training_images/86016/"
    arr = load_image(path+"Test image.jpg")
    best_inds = nsga_ii(
        arr, constraints=(4, 41), pop_size=75, g_max=25)

    for i, best_ind in enumerate(best_inds):
        print("Individual number:", i)
        print("Number of segmentations:", best_ind.num_segments)
        print("Edge value:", -best_ind.edge_value)
        print("Connectivity:", best_ind.connectivity)
        print("Deviation:", best_ind.deviation)
        print()
    
        save_black_white(path+str(i), arr, best_ind.segments, isEdge)
    
    for i, best_ind in enumerate(best_inds):
    
        save_segmented(path+"_"+str(i), arr, best_ind.segments, isEdge)
        