import numpy as np
import time
from random import random, randint

from chromosome import (
    Chromosome, 
    crossover,
    elitist_replacement,
    mutate,
    parent_selection,
    sort_by_fitness
)
#from nsga_chromosome import nd_sort, crowding_distance
from utils import isEdge
from image import load_image, save_black_white, save_segmented

def ga(pixel_arr, constraints, pop_size=100, g_max=40, p_c=0.6, p_m=0.4, verbose=True):

    start_time = time.time()
    num_rows, num_cols = pixel_arr.shape[0], pixel_arr.shape[1]
    population = [Chromosome(pixel_arr) for i in range(pop_size)]
    avg_fitness = sum(c.fitness for c in population)
    best_ind = max(population, key=lambda c: c.fitness)
    
    if verbose:
        print("Generation 0:")
        print("Best fitness:", best_ind.fitness, "Avg fitness:", avg_fitness)
        print("Number of segmentations:", max(best_ind.segments))
        print("Edge value:", best_ind.edge_value)
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

        population = elitist_replacement(population, offspring)

        avg_fitness = sum(c.fitness for c in population)
        best_ind = max(population, key=lambda c: c.fitness)
        
        if verbose:
            print(f"Generation {g}:")
            print("Best fitness:", best_ind.fitness, "Avg fitness:", avg_fitness)
            print("Number of segmentations:", max(best_ind.segments))
            print("Edge value:", best_ind.edge_value)
            print("Connectivity:", best_ind.connectivity)
            print("Deviation:", best_ind.deviation)
            print(f"Time elapsed: {time.time() - start_time}\n", flush=True)
    
    return best_ind

if __name__ == "__main__":
    path = "training_images/86016/"
    arr = load_image(path+"Test image.jpg")
    best_ind = ga(
        arr, constraints=(4, 41), pop_size=75, g_max=25
        )

    print("Number of segmentations:", max(best_ind.segments))
    print("Edge value:", best_ind.edge_value)
    print("Connectivity:", best_ind.connectivity)
    print("Deviation:", best_ind.deviation)
    
   

    save_black_white(path+"simple_ga", arr, best_ind.segments, isEdge)
    save_segmented(path+"simple_ga_green", arr, best_ind.segments, isEdge)