import numpy as np
import pandas as pd

np.random.seed(1)

df = pd.read_csv("dataset.csv")

dataset = df.to_numpy()
x_features = dataset[:, :-1]  # Slice last element from every row
y_values = dataset[:, -1]    # Get only the last element from every row


def generate_population(bitstring_length, population_size):
    population = []
    for i in range(population_size):
        bitstring = ['1' if np.random.rand(
        ) >= 0.5 else '0' for i in range(bitstring_length)]  # Generate random bitstring of '1's and '0's

        population.append("".join(bitstring))  # Append bitstring to population

    return population


def parent_selection(fitness_values, population, number_of_parents):

    # Roulette wheel selection of parents
    # Number of parents should be an even number
    pop_to_select_from = population[:]
    parents = []
    for i in range(number_of_parents):
        total_fitness = np.sum(fitness_values)
        # Selected is a score between 0 and total_fitness
        selected = np.random.rand() * total_fitness
        partial_sum = 0
        for j in range(len(pop_to_select_from)):
            partial_sum += fitness_values[j]  # Sum up fitness values
            if partial_sum >= selected:
                # Remove parent to prevent reselection and add to parents list
                parents.append(pop_to_select_from.pop(j))

    return parents


def generate_offspring(parents, mutation_prob):
    # Generates offspring through crossover
    # Every bit in child-bitstring has small chance of mutation

    np.random.shuffle(parents)  # Randomize which individuals become parents
    for i in range(0, len(parents), 2):
