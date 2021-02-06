import numpy as np
import pandas as pd
import operator
from LinReg import LinReg

np.random.seed(1)


df = pd.read_csv("dataset.csv")

dataset = df.to_numpy()
x_features = dataset[:, :-1]  # Slice last element from every row
y_values = dataset[:, -1]    # Get only the last element from every row


POP_SIZE = 120
BITSTRING_LENGTH = 15
MUTATION_PROB = 0.05
CROSSOVER_PROB = 0.7
MAX_GENERATIONS = 600


class Individual():
    def __init__(self, bitstring):
        self.bitstring = bitstring
        self.phenotype = None
        self.fitness = None
        self.RMSE = None


def generate_initial_population(bitstring_length, population_size, fitness_func):
    population = []
    for i in range(population_size):
        bitstring = ['1' if np.random.rand(
        ) >= 0.5 else '0' for i in range(bitstring_length)]  # Generate random bitstring of '1's and '0's

        # Assign bitstring/genotype to an individual and calculate fitness
        individual = Individual("".join(bitstring))
        fitness_func(individual)

        # Add individual to population
        population.append(individual)

    return population


def parent_selection(population, fitness_sum, population_size):
    # Calculate random threshold for partial sum of fitness values
    selected = np.random.rand() * fitness_sum
    partial_sum = 0
    for i in range(population_size):
        partial_sum += population[i].fitness  # Sum up fitness values
        if partial_sum >= selected:
            # Return first individual who exceeded partial sum threshold
            return population[i]


def crossover(parent_1, parent_2, crossover_prob):

    # Could also randomise if crossover happens all together
    if np.random.rand() <= crossover_prob:
        # Calculate random crossover point and generate two new bitstrings
        bitstring_length = len(parent_1.bitstring)
        crossover_point = np.random.randint(0, bitstring_length)
        # print(parent_1.bitstring, parent_2.bitstring)
        child_1 = parent_1.bitstring[:crossover_point] + \
            parent_2.bitstring[crossover_point:]
        child_2 = parent_2.bitstring[:crossover_point] + \
            parent_1.bitstring[crossover_point:]
        # print(child_1, child_2)

        return Individual(child_1), Individual(child_2)

    else:
        # If no crossover occurs, we generate children who have the same genes as their parents
        return Individual(parent_1.bitstring), Individual(parent_2.bitstring)


def mutation(individual, mutation_prob):
    # To flip bits we need a mutable object, so we convert to list
    bitstring = list(individual.bitstring)
    for i in range(len(bitstring)):
        if np.random.rand() <= mutation_prob:
            bitstring[i] = '1' if bitstring[i] == '0' else '0'  # Swap bits

    individual.bitstring = "".join(bitstring)


def generate_new_population(old_population, mutation_prob, crossover_prob, population_size, fitness_func, fitness_sum):
    # Generates offspring through crossover
    # Every bit in child-bitstring has small chance of mutation
    # Assumes that population_size is an even number

    new_population = []

    for i in range(0, population_size, 2):

        # Select two parents
        mate_1 = parent_selection(old_population, fitness_sum, population_size)
        mate_2 = parent_selection(old_population, fitness_sum, population_size)

        # Generate children from parents and apply possible mutation
        child_1, child_2 = crossover(mate_1, mate_2, crossover_prob)
        mutation(child_1, mutation_prob)
        mutation(child_2, mutation_prob)

        # Evaluate fitness of the children and add to new population
        fitness_func(child_1)
        fitness_func(child_2)
        new_population.append(child_1)
        new_population.append(child_2)

    return new_population


def survivor_selection_elitism(old_population, new_population, pop_size):
    # Survivor selection. Sort populations by fitness.
    # The 95 % fittest in the new pop and 5 % fittest in old pop survive

    sorted_old_pop = sorted(
        old_population, key=operator.attrgetter('fitness'))
    sorted_new_pop = sorted(
        new_population, key=operator.attrgetter('fitness'))

    cutoff_old = int(pop_size * 0.95)  # Elitism
    cutoff_new = pop_size - cutoff_old

    return (sorted_new_pop[cutoff_new:] + sorted_old_pop[cutoff_old:])


def survivor_selection_fitness(old_population, new_population, pop_size):

    sorted_population = sorted(
        old_population + new_population, key=operator.attrgetter('fitness'))

    return sorted_population[pop_size:]


def genetic_algorithm(pop_size, bitstring_length, prob_m, prob_c, f, g_max, survivor_func):
    generation = 0
    fitness_sum = 0
    old_population = generate_initial_population(bitstring_length, pop_size, f)
    # fitness_diff(old_population)
    population_list = []
    population_list.append(old_population)
    while generation < g_max:
        # Calculate fitness_sum for current population
        for individual in old_population:
            fitness_sum += individual.fitness
        generation += 1

        np.random.shuffle(old_population)

        new_population = generate_new_population(
            old_population, prob_m, prob_c, pop_size, f, fitness_sum)
        # Replacement - can also implement survivor selection
        surviving_population = survivor_func(
            old_population, new_population, pop_size)

        old_population = surviving_population
        # fitness_diff(old_population)
        population_list.append(old_population)

        # Reset fitness_sum
        fitness_sum = 0

    return population_list


def fitness_diff(population):
    least_fit_ind = min(population, key=operator.attrgetter("fitness"))
    correction = least_fit_ind.fitness

    #print("NEW GENERATION least fit:", correction)
    for individual in population:
        #print("before diff:", individual.fitness)
        individual.fitness -= correction
        #print("after diff:", individual.fitness)


def sine_fitness(individual: Individual):

    # Max value of phenotype is 128 = 2^7
    scaling_factor = 2 ** (len(individual.bitstring) - 7)
    # Get phenotype by converting from bitstring to decimal number and divide by scaling factor
    individual.phenotype = int(individual.bitstring, 2) / scaling_factor
    fitness = np.sin(individual.phenotype)

    # np.sin() returns a number between -1 and 1, but we need positive values for
    # roulette-wheel parent selection to work. Scale by adding 1 and dividing by 2

    individual.fitness = (fitness + 1) / 2


def linreg_fitness(individual):
    model = LinReg()
    features = model.get_columns(x_features, individual.bitstring)
    # Set phenotype as number of features and set fitness
    individual.phenotype = model.train(features, y_values)
    individual.RMSE = model.get_fitness(features, y_values)
    # We want to minimize error, but the algorithm tries to maximize fitness
    # Define fitness as 1 - RMSE. Smaller RMSE -> Higher fitness
    individual.fitness = 1 - individual.RMSE


"""
generations = genetic_algorithm(
    POP_SIZE,
    BITSTRING_LENGTH,
    MUTATION_PROB,
    CROSSOVER_PROB,
    sine_fitness, 
    MAX_GENERATIONS,
    survivor_selection_fitness)

for gen in generations:
    
   for ind in gen:
        print("Phenotype score of individual:", ind.phenotype)
    
    avg_score = np.mean([i.fitness for i in gen])
    print(avg_score)
"""

generations = genetic_algorithm(
    POP_SIZE,
    x_features.shape[1],
    MUTATION_PROB,
    CROSSOVER_PROB,
    linreg_fitness,
    MAX_GENERATIONS,
    survivor_selection_fitness)

model = LinReg()
print("RMSE without feature selection:",
      model.get_fitness(x_features, y_values))
for gen in generations:
    sorted_gen = sorted(gen, key=operator.attrgetter('RMSE'))
    print("Fittest individual of generation:", sorted_gen[0].RMSE)
    #print("RMSE score of individual:", 1 - ind.fitness)
    avg_score = np.mean([i.RMSE for i in gen])
    print("Average score per generation:", avg_score)
