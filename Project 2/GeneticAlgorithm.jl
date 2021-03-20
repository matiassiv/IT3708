module GeneticAlgorithm

export generate_initial_generation, GA,  binary_tournament_selection
#push!(LOAD_PATH, pwd())
include("MDVRProblem.jl")
#include("Utils.jl")
using .MDVRProblem: Chromosome, Depot, init_random_chromosome, init_chromosome, route_scheduler!
using ..Utils: ProblemInstance
using Random

struct GAState
    pop_size::Int
    population::Vector{Chromosome}
    pop_fitness::Float64
    fittest_ind::Chromosome
end

# First create GA without interdepot mutation

function GA(
    pop_size::Int,
    max_generations::Int,
    problem_params::ProblemInstance,
    crossover_rate::Float64=0.6,
    mutation_rate::Float64=0.05,
    inter_depot_rate::Float64=0.01,
    selection::Function=binary_tournament_selection,
    crossover::Function=simple_crossover,
    mutation::Function=(x,y,z)->x,
    replacement::Function=simple_replacement,
    )::Chromosome   

    population_log = Vector{GAState}(undef, max_generations+1)
    generation = generate_initial_generation(pop_size, problem_params)
    best_individual = generation.fittest_ind
    population_log[1] = generation

    for g = 1:max_generations
        #println(generation.pop_fitness / pop_size)
        selected = selection(generation)
        
        offspring = crossover(selected, crossover_rate, problem_params)
        #mutated_offspring = mutation(offspring, mutation_rate, inter_depot_rate)
        
        generation = replacement(generation, offspring)

        if generation.fittest_ind.fitness < best_individual.fitness
            best_individual = generation.fittest_ind
        end 

        population_log[g+1] = generation
        
    end
    return best_individual

end


function generate_initial_generation(pop_size::Int, params::ProblemInstance)::GAState
    population = Vector{Chromosome}(undef, pop_size)

    population[1] = init_random_chromosome(
            params.num_depots,
            params.num_customers,
            params.max_vehicles,
            params.depot_assignments,
            params.depot_info,
            params.customer_info,
            params.distances
        )
    pop_fitness = population[1].fitness
    fittest_ind = population[1]
    fittest_score = population[1].fitness
    for i = 2:pop_size

        population[i] = init_random_chromosome(
            params.num_depots,
            params.num_customers,
            params.max_vehicles,
            params.depot_assignments,
            params.depot_info,
            params.customer_info,
            params.distances
        )
        pop_fitness += population[i].fitness
        if population[i].fitness < fittest_score
            fittest_score = population[i].fitness
            fittest_ind = population[i]
            #println(fittest_score)
        end
    end

    return GAState(pop_size, population, pop_fitness, fittest_ind)
end

function binary_tournament_selection(generation::GAState)::Vector{Chromosome}
    selected = Vector{Chromosome}(undef, generation.pop_size)
    
    # Want to generate pop_size parents
    for i = 1 : generation.pop_size
        # Get two individuals
        t1_i = Random.rand(1:generation.pop_size)
        t2_i = Random.rand(1:generation.pop_size)
        t1 = generation.population[t1_i]
        t2 = generation.population[t2_i]
        r = Random.rand()

        # If r is less than 0.8 then the best individual is selected
        if r < 0.8
            selected[i] = t1.fitness < t2.fitness ? t1 : t2
        # If r is larger than 0.8 then a coin flip decides the parent
        else
            coin_flip = Random.rand()
            selected[i] = coin_flip >= 0.5 ? t1 : t2
        end

    end
    #@assert length(selected) == length(generation.population)
    return selected
end

function simple_crossover(
    selected::Vector{Chromosome},
    crossover_rate::Float64,
    problem_params::ProblemInstance
    )::Vector{Chromosome}
    offspring = similar(selected)
    for i = 1:2:length(selected)
        p1 = selected[i]
        p2 = selected[i+1]

        c = Random.rand()
        if c <= crossover_rate
            # Select random depot to crossover
            depot = Random.rand(1:problem_params.num_depots)

            # Select random route within that depot to crossover
            num_r_p1 = Random.rand(1:p1.depots[depot].num_routes)
            num_r_p2 = Random.rand(1:p2.depots[depot].num_routes)
            r_p1 = p1.depots[depot].routes[num_r_p1]
            r_p2 = p2.depots[depot].routes[num_r_p2]
            
            #@assert length(r_p1) > 0
            #@assert length(r_p2) > 0

            
            # Delete elements from the other parents route from the route encoding
            route_encoding_1 = Dict{Int, Vector{Int}}()
            route_encoding_2 = Dict{Int, Vector{Int}}()
            for i = 1:problem_params.num_depots
                route_encoding_1[i] = setdiff(p1.depots[i].route_encoding, r_p2)
                route_encoding_2[i] = setdiff(p2.depots[i].route_encoding, r_p1)
            end
        

            find_best_feasible!(r_p2, route_encoding_1, problem_params, depot)
            find_best_feasible!(r_p1, route_encoding_2, problem_params, depot)
            
            offspring[i] = init_chromosome(
                problem_params.num_depots,
                problem_params.num_customers,
                problem_params.max_vehicles,
                route_encoding_1,
                problem_params.depot_info,
                problem_params.customer_info,
                problem_params.distances
            )
            offspring[i+1] = init_chromosome(
                problem_params.num_depots,
                problem_params.num_customers,
                problem_params.max_vehicles,
                route_encoding_2,
                problem_params.depot_info,
                problem_params.customer_info,
                problem_params.distances
            )

        
        else
            offspring[i] = p1
            offspring[i+1] = p2
        end 
    end
    return offspring

end

function simple_replacement(generation::GAState, offspring::Vector{Chromosome})::GAState
    sort!(generation.population, by=x->x.fitness)
    sort!(offspring, by=x->x.fitness)
    pop_size = length(offspring)
    population = Vector{Chromosome}(undef, pop_size)

    population[1] = generation.population[1]
    pop_fitness = population[1].fitness
    fittest_ind = population[1]
    fittest_score = population[1].fitness
    elitism = 4
    for i = 2:pop_size
        if i <= elitism
            population[i] = generation.population[i]
        else
            population[i] = offspring[i-elitism]
        end
        pop_fitness += population[i].fitness
        if population[i].fitness < fittest_score
            fittest_score = population[i].fitness
            fittest_ind = population[i]
            #println(fittest_score)
        end
    end

    return GAState(pop_size, population, pop_fitness, fittest_ind)


end

function find_best_feasible!(
    parent_route::Vector{Int}, 
    route_encoding::Dict{Int, Vector{Int}}, 
    problem_params::ProblemInstance, 
    depot::Int
    )
    for customer in parent_route
        depot_id = depot
        # Try to insert customer at all the different places
        # Randomise depot if it's a borderline customer
        if haskey(problem_params.borderline_customers, customer)
            depot_id = Random.rand(problem_params.borderline_customers[customer])
        end

        #@assert customer in values(problem_params.depot_assignments[depot_id]) || haskey(problem_params.borderline_customers, customer)

        best_fitness = Inf
        # Holds intermediate results has shape [(depot_id fitness, route_encoding)]
        best_encoding = Vector{Int}(undef, length(route_encoding[depot_id])+1)
        for i = 1:length(route_encoding[depot_id])+1
            
            # Copy the route_encoding for the "crossovered" depot_id 
            #enc = route_encoding[depot_id]
            # Add the customer to copied array
            insert!(route_encoding[depot_id], i, customer)

            # Create Depot object for the route scheduler
            d = Depot(
                route_encoding[depot_id],
                1, 
                Dict(),
                Vector(),
                Vector(), 
                problem_params.depot_info[depot_id][3], 
                problem_params.depot_info[depot_id][4],
                problem_params.max_vehicles, 
                depot_id
            )
            fitness = route_scheduler!(d, problem_params.customer_info, problem_params.distances, problem_params.num_depots)
            # Add to intermediate results - the i lets min function differentiate
            if fitness < best_fitness
                best_fitness = fitness
                best_encoding = copy(route_encoding[depot_id])
            end
            deleteat!(route_encoding[depot_id], i)
        end
        route_encoding[depot_id] = best_encoding
    end
end

end

