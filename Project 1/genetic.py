import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from LinReg import LinReg

np.random.seed(1)


df = pd.read_csv("dataset.csv")

dataset = df.to_numpy()
x_features = dataset[:, :-1]  # Slice last element from every row
y_values = dataset[:, -1]    # Get only the last element from every row


POP_SIZE = 100
BITSTRING_LENGTH = 15
MUTATION_PROB = 0.02
CROSSOVER_PROB = 0.7
SINE_WAVE_MAX_GEN = 7
MAX_GENERATIONS = 50


class Individual():
    def __init__(self, bitstring, parents=[]):
        self.bitstring = bitstring
        self.phenotype = None
        self.fitness = None
        self.RMSE = None
        self.parents = parents


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


def parent_selection_roulette(population, fitness_sum, population_size):
    # Calculate random threshold for partial sum of fitness values
    selected = np.random.rand() * fitness_sum
    partial_sum = 0
    for i in range(population_size):
        partial_sum += population[i].fitness  # Sum up fitness values
        if partial_sum >= selected:
            # Return first individual who exceeded partial sum threshold
            return population[i]


def parent_selection_tournament(population, fitness_sum=0, population_size=100):

    # Get two random individuals from population
    random_1 = np.random.randint(0, population_size)
    random_2 = np.random.randint(0, population_size)
    random_3 = np.random.randint(0, population_size)
    # random_4 = np.random.randint(0, population_size)

    t1 = population[random_1]
    t2 = population[random_2]
    t3 = population[random_3]
    # t4 = population[random_4]

    # Compare fitness to select parent

    return max(t1, t2, t3, key=operator.attrgetter("fitness"))


def crossover(parent_1, parent_2, crossover_prob):

    # Check if crossover happens all together
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

        return Individual(child_1, [parent_1, parent_2]), Individual(child_2, [parent_1, parent_2])

    else:
        # If no crossover occurs, we generate children who have the same genes as their parents
        return Individual(parent_1.bitstring, [parent_1, parent_2]), Individual(parent_2.bitstring, [parent_1, parent_2])


def mutation(individual, mutation_prob):
    # To flip bits we need a mutable object, so we convert to list
    bitstring = list(individual.bitstring)
    for i in range(len(bitstring)):
        if np.random.rand() <= mutation_prob:
            bitstring[i] = '1' if bitstring[i] == '0' else '0'  # Swap bits

    individual.bitstring = "".join(bitstring)


def generate_new_population(
        old_population,
        mutation_prob,
        crossover_prob,
        population_size,
        fitness_func,
        fitness_sum,
        parent_func):
    # Generates offspring through crossover
    # Every bit in child-bitstring has small chance of mutation
    # Assumes that population_size is an even number

    new_population = []

    for i in range(0, population_size, 2):

        # Select two parents
        mate_1 = parent_func(old_population, fitness_sum, population_size)
        mate_2 = parent_func(old_population, fitness_sum, population_size)

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


def survivor_selection_age(old_population, new_population, pop_size):
    # Survivor selection which only depends on age

    return new_population


def survivor_selection_elitism(old_population, new_population, pop_size):
    # Survivor selection. Sort populations by fitness.
    # The x % fittest in the new pop and 1 -x % fittest in old pop survive
    # Normal to select high x %

    sorted_old_pop = sorted(
        old_population, key=operator.attrgetter('fitness'))
    sorted_new_pop = sorted(
        new_population, key=operator.attrgetter('fitness'))

    cutoff_old = int(pop_size * 0.7)  # Elitism
    cutoff_new = pop_size - cutoff_old

    return (sorted_new_pop[cutoff_new:] + sorted_old_pop[cutoff_old:])


def survivor_selection_fitness(old_population, new_population, pop_size):

    sorted_population = sorted(
        old_population + new_population, key=operator.attrgetter('fitness'))

    return sorted_population[pop_size:]


def deterministic_crowding(old_population, pop_size, crossover_prob, mutation_prob, fitness_func, generation):
    surviving_population = []
    for i in range(0, pop_size, 2):

        # Select two parents at random (old population is shuffled before function)
        # if generation < 50:
        mate_1 = old_population[i]
        mate_2 = old_population[i+1]

        # else:
        # Tournament-style parent selection
        #   mate_1 = parent_selection_tournament(
        #     old_population, population_size=pop_size)
        #  mate_2 = parent_selection_tournament(
        #    old_population, population_size=pop_size)

        # Generate children from parents and apply possible mutation
        child_1, child_2 = crossover(mate_1, mate_2, crossover_prob)
        mutation(child_1, mutation_prob)
        mutation(child_2, mutation_prob)

        # Evaluate fitness of the children and add to new population
        fitness_func(child_1)
        fitness_func(child_2)

        if (hamming_distance(mate_1, child_1) + hamming_distance(mate_2, child_2)) < \
            (hamming_distance(mate_1, child_2) +
             hamming_distance(mate_2, child_1)):
            s1 = child_1 if child_1.fitness >= mate_1.fitness else mate_1
            s2 = child_2 if child_2.fitness >= mate_2.fitness else mate_2
            surviving_population.append(s1)
            surviving_population.append(s2)

        else:
            s1 = child_2 if child_2.fitness >= mate_1.fitness else mate_1
            s2 = child_1 if child_1.fitness >= mate_2.fitness else mate_2
            surviving_population.append(s1)
            surviving_population.append(s2)

    return surviving_population


def genetic_algorithm(
        pop_size,
        bitstring_length,
        prob_m,
        prob_c,
        f,
        g_max,
        survivor_func=survivor_selection_fitness,
        parent_func=parent_selection_tournament,
        crowding=False):

    generation = 0
    maximum = 0  # To implement early stopping in case of convergence
    early_stop_counter = 0
    fitness_sum = 0
    old_population = generate_initial_population(bitstring_length, pop_size, f)
    population_list = []
    population_list.append(old_population)

    while generation < g_max:

        generation += 1

        # Calculate fitness_sum for current population
        for individual in old_population:
            fitness_sum += individual.fitness

        # print("Fitness_sum:", fitness_sum)

        np.random.shuffle(old_population)

        if crowding:
            surviving_population = deterministic_crowding(
                old_population, pop_size, prob_c, prob_m, f, generation)

        else:
            new_population = generate_new_population(
                old_population, prob_m, prob_c, pop_size, f, fitness_sum, parent_func)
            # Replacement step
            surviving_population = survivor_func(
                old_population, new_population, pop_size)

        old_population = surviving_population
        population_list.append(old_population)

        # Calculate if early stop is necessary (if converged)
        if fitness_sum > maximum:
            maximum = fitness_sum
            early_stop_counter = 0
            fitness_sum = 0
        else:
            early_stop_counter += 1
            fitness_sum = 0

        if early_stop_counter >= 20:
            break

    return population_list


def sine_fitness(individual: Individual):

    # Max value of phenotype is 128 = 2^7
    scaling_factor = 2 ** (len(individual.bitstring) - 7)
    # Get phenotype by converting from bitstring to decimal number and divide by scaling factor
    individual.phenotype = int(individual.bitstring, 2) / scaling_factor
    fitness = np.sin(individual.phenotype)

    # np.sin() returns a number between -1 and 1, but we need positive values for
    # roulette-wheel parent selection to work. Scale by adding 1 and dividing by 2

    individual.fitness = (fitness + 1)


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
def linreg_fitness_diversity(individual):
    model = LinReg()
    features = model.get_columns(x_features, individual.bitstring)
    # Set phenotype as number of features and set fitness
    individual.phenotype = model.train(features, y_values)
    individual.RMSE = model.get_fitness(features, y_values)
    # We want to minimize error, but the algorithm tries to maximize fitness
    # Define fitness as 1 - RMSE. Smaller RMSE -> Higher fitness
    k = 0.0001
    individual.fitness = 1 - individual.RMSE + k * \
        np.sum([hamming_distance(individual, other) for other in population])

    print(individual.fitness)
"""


def calc_entropy(generation, bitstring_length):
    entropy = 0
    for i in range(bitstring_length):
        bit_counter = 0
        for ind in generation:
            if ind.bitstring[i] == '1':
                bit_counter += 1

        # Need to check that there actually is a '1'-bit, as log2(0) is not defined
        if bit_counter > 0:
            bit_prob = bit_counter / len(generation)
            entropy += bit_prob * np.log2(bit_prob)

    return -(entropy)


def hamming_distance(ind_1, ind_2):
    distance = 0
    for i in range(len(ind_1.bitstring)):
        if ind_1.bitstring[i] != ind_2.bitstring[i]:
            distance += 1

    return distance


if __name__ == "__main__":

    generations = genetic_algorithm(
        POP_SIZE,
        BITSTRING_LENGTH,
        MUTATION_PROB,
        CROSSOVER_PROB,
        sine_fitness,
        SINE_WAVE_MAX_GEN,
        survivor_selection_fitness)

    x_values = np.arange(0, 128, 0.1)
    sine = np.sin(x_values)
    for i in range(len(generations)):
        phenotypes = []
        fitnesses = []
        for ind in generations[i]:
            phenotypes.append(ind.phenotype)
            fitnesses.append(ind.fitness - 1)
        avg_score = np.mean([i.fitness - 1 for i in generations[i]])
        plt.plot(x_values, sine, label="sin(x)")
        plt.scatter(phenotypes, fitnesses, color='red')
        plt.title("Generation " + str(i))
        plt.legend()
        plt.savefig("task_1e_sin(x)_gen_"+str(i)+".png")
        plt.clf()

    generations = genetic_algorithm(
        POP_SIZE,
        BITSTRING_LENGTH,
        MUTATION_PROB,
        CROSSOVER_PROB,
        sine_fitness,
        SINE_WAVE_MAX_GEN*3,
        crowding=True)

    x_values = np.arange(0, 128, 0.1)
    sine = np.sin(x_values)
    for i in range(len(generations)):
        phenotypes = []
        fitnesses = []
        for ind in generations[i]:
            phenotypes.append(ind.phenotype)
            fitnesses.append(ind.fitness - 1)
        avg_score = np.mean([i.fitness - 1 for i in generations[i]])
        plt.plot(x_values, sine, label="sin(x)")
        plt.scatter(phenotypes, fitnesses, color='red')
        plt.title("Generation " + str(i))
        plt.legend()
        plt.savefig("task_1g_sin(x)_crowding_gen_"+str(i)+".png")
        plt.clf()

    generations = genetic_algorithm(
        POP_SIZE,
        x_features.shape[1],
        MUTATION_PROB,
        CROSSOVER_PROB,
        linreg_fitness,
        MAX_GENERATIONS,
        survivor_selection_elitism)

    model = LinReg()
    fitness_vals = [model.get_fitness(x_features, y_values)]
    entropy_vals = []
    entropy_x_vals = np.arange(0, len(generations), 1)
    x_labels = ["Base"]
    # print("RMSE without feature selection:",
    # fitness_vals[0])
    for i, gen in enumerate(generations):

        # sorted_gen = sorted(gen, key=operator.attrgetter('RMSE'))
        # print("Fittest individual of generation:", sorted_gen[0].RMSE)

        avg_RMSE = np.mean([i.RMSE for i in gen])
        # print("Average RMSE per generation:", avg_RMSE)

        entropy_vals.append(calc_entropy(gen, x_features.shape[1]))

        # Get avg fitness of every 20th generation for plot
        if i % 20 == 0:
            fitness_vals.append(avg_RMSE)
            x_labels.append(str(i))

    print("Number of generations:", len(generations))
    print("Final generation: ", np.mean([i.RMSE for i in generations[-1]]))
    sorted_gen = sorted(generations[-1], key=operator.attrgetter('RMSE'))
    print("Fittest individual of final generation:", sorted_gen[0].RMSE)
    y_pos = np.arange(len(x_labels))
    plt.ylim([0.12, 0.14])
    plt.bar(y_pos, fitness_vals, align="center", alpha=0.5)
    plt.xticks(y_pos, x_labels)
    plt.ylabel("Mean RMSE")
    plt.title("Mean RMSE for selected generations")
    plt.savefig("task_1f_avg_RMSE.png")
    plt.clf()

    generations = genetic_algorithm(
        POP_SIZE,
        x_features.shape[1],
        MUTATION_PROB,
        CROSSOVER_PROB,
        linreg_fitness,
        MAX_GENERATIONS,
        crowding=True)

    model = LinReg()
    fitness_vals = [model.get_fitness(x_features, y_values)]
    x_labels = ["Base"]
    entropy_vals_crowding = []
    entropy_x_vals_crowding = np.arange(0, len(generations), 1)
    print("\nCROWDING\n")
    for i, gen in enumerate(generations):

        # sorted_gen = sorted(gen, key=operator.attrgetter('RMSE'))
        # print("Fittest individual of generation:", sorted_gen[0].RMSE)

        avg_RMSE = np.mean([i.RMSE for i in gen])
        # print("Average RMSE per generation:", avg_RMSE)

        entropy_vals_crowding.append(calc_entropy(gen, x_features.shape[1]))

        # Get avg fitness of every 20th generation for plot
        if i % 15 == 0:
            fitness_vals.append(avg_RMSE)
            x_labels.append(str(i))

    print("Number of generations:", len(generations))
    print("RMSE without feature selection:", fitness_vals[0])
    print("Final generation: ", np.mean([i.RMSE for i in generations[-1]]))
    sorted_gen = sorted(generations[-1], key=operator.attrgetter('RMSE'))
    print("Fittest individual of final generation:", sorted_gen[0].RMSE)
    y_pos = np.arange(len(x_labels))
    plt.ylim([0.12, 0.15])
    plt.bar(y_pos, fitness_vals, align="center", alpha=0.5)
    plt.xticks(y_pos, x_labels)
    plt.ylabel("Mean RMSE")
    plt.title("Mean RMSE for selected generations")
    plt.savefig("task_1g_avg_RMSE.png")
    plt.clf()

    plt.plot(entropy_x_vals, entropy_vals, label="Without crowding")
    plt.plot(entropy_x_vals_crowding, entropy_vals_crowding, label="Crowding")
    plt.title("Entropy change over generations with crowding")
    plt.legend()
    plt.savefig("task_1g_entropy_crowding.png")
    plt.clf()

"""
Theory questions:
Niching is uses of techniques to make populations in evolutionary algorithms converge towards multiple
"interesting" optima. In standard versions, the population will tend towards converging to the same solution,
which may not be the global optimum. Niching is a way to make sure more of these local optima are considered.
This can be beneficial in a project like this, where the search space is very large. By considering more optima,
through diversification of the population it will be more likely to find a better global solution, than if 
the algorithm races up the first optimum it finds.

Genotype in this project is simply the bitstring corresponding to which features the linreg model should consider.
The phenotype is the actual value passed into the fitness function, which in this case would be the trained weights
from the linreg model. The genotype decides which columns the model should train on, resulting in the phenotype which is
the learnt weights from training.

"""
