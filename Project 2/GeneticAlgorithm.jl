module GeneticAlgorithm

export generate_initial_generation, GA,  binary_tournament_selection
#push!(LOAD_PATH, pwd())
include("MDVRProblem.jl")
include("FileParser.jl")
#include("Utils.jl")
using .MDVRProblem: Chromosome, Depot, init_random_chromosome, init_chromosome, route_scheduler!, calculate_chromosome_fitness!
using .FileParser: create_solution_file
using ..Utils: ProblemInstance
using Random
using StatsBase

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
    benchmark::Int=300,
    crossover_rate::Float64=0.7,
    mutation_rate::Float64=0.09,
    selection::Function=binary_tournament_selection,
    crossover::Function=simple_crossover,
    mutation::Function=simple_mutation!,
    replacement::Function=simple_replacement,
    )::Chromosome   

    early_stop = 0
    early_benchmark = true
    #population_log = Vector{GAState}()
    generation = generate_initial_generation(pop_size, problem_params)
    best_individual = generation.fittest_ind
    #push!(population_log, generation)
    

    for g = 1:max_generations
        selected = selection(generation)
        
        offspring = crossover(selected, crossover_rate, problem_params)
        mutation(offspring, mutation_rate, pop_size, problem_params)
        
        generation = replacement(generation, offspring)
        #push!(population_log, generation)

        
        if generation.fittest_ind.fitness < best_individual.fitness
            best_individual = deepcopy(generation.fittest_ind)
            early_stop = 0
        else
            early_stop += 1
        end
        if g % 15 == 0
            println("\nGen ", g, "\nBest: ", best_individual.fitness, "  Avg: ", generation.pop_fitness/pop_size)
        end
                    
        if early_stop >= 60
            println("Early stop kicked in at generation: ", g)
            break
        end
        
        if best_individual.fitness <= benchmark*1.05
            println("Benchmark reached at generation: ", g)
            break
        end

        if early_benchmark && best_individual.fitness <= benchmark*1.1
            create_solution_file(best_individual, "temp_benchmark.txt")
            early_benchmark = false
        end
        
        
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
            
            r = Random.rand()
            if r < 0.3*problem_params.num_depots # It's very costly to run when routes are long
                find_best_feasible!(r_p2, route_encoding_1, problem_params, depot)
                find_best_feasible!(r_p1, route_encoding_2, problem_params, depot)
            else
                find_random_best_feasible!(r_p2, route_encoding_1, problem_params, depot)
                find_random_best_feasible!(r_p1, route_encoding_2, problem_params, depot)
            end
            
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
    elitism = 25
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

function find_random_best_feasible!(
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
        num_insertions = (length(route_encoding[depot_id])+1) ÷ 3
        insertion_points = StatsBase.sample(1:length(route_encoding[depot_id])+1, num_insertions, replace=false)
        best_fitness = Inf
        # Holds intermediate results has shape [(depot_id fitness, route_encoding)]
        best_encoding = Vector{Int}(undef, 3)
        for i in insertion_points
            
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

function simple_mutation!(offspring::Vector{Chromosome}, mutation_rate::Float64, pop_size::Int, problem_params::ProblemInstance)
    for i = 1:pop_size
        m = Random.rand()
        if m <= mutation_rate
            
            mut_type = Random.rand()
            if mut_type < 0.15
                reversal!(offspring[i], problem_params)
                #=
            elseif mut_type >= 0.95
                swap!(offspring[i], problem_params)
            =#
            else
                offspring[i] = single_rerouting(offspring[i], problem_params)
            end
        
        end
    end

end

function reversal!(c::Chromosome, problem_params::ProblemInstance)
    d = Random.rand(1:c.num_depots)
    depot = c.depots[d]
    cutpoint_1 = Random.rand(1:length(depot.route_encoding))
    cutpoint_2 = Random.rand(1:length(depot.route_encoding))
    while true
        if cutpoint_1 != cutpoint_2
            break
        end
        cutpoint_2 = Random.rand(1:length(depot.route_encoding))
    end
    if cutpoint_1 < cutpoint_2
        reverse!(depot.route_encoding, cutpoint_1, cutpoint_2)
    else
        reverse!(depot.route_encoding, cutpoint_2, cutpoint_1)
    end
    new_depot = Depot(
        depot.route_encoding, 
        1, 
        Dict(),
        Vector(),
        Vector(), 
        problem_params.depot_info[d][3], 
        problem_params.depot_info[d][4],
        problem_params.max_vehicles, 
        d
        )
    route_scheduler!(new_depot, problem_params.customer_info, problem_params.distances, problem_params.num_depots)
    c.depots[d] = new_depot
    calculate_chromosome_fitness!(c)
end


function single_rerouting(c::Chromosome, problem_params::ProblemInstance)
    depot_id = Random.rand(1:length(c.depots))
    customer = Random.rand(c.depots[depot_id].route_encoding)

    route_encoding = Dict{Int, Vector{Int}}()
    for i = 1:problem_params.num_depots
        if i == depot_id
            route_encoding[i] = filter(x->x != customer, c.depots[i].route_encoding)
        else
            route_encoding[i] = c.depots[i].route_encoding
        end
    end
    find_best_feasible!([customer], route_encoding, problem_params, depot_id)
    
    c1 = init_chromosome(
        problem_params.num_depots,
        problem_params.num_customers,
        problem_params.max_vehicles,
        route_encoding,
        problem_params.depot_info,
        problem_params.customer_info,
        problem_params.distances
    )

    return c1
end

function swap!(c::Chromosome, problem_params::ProblemInstance)
    d = Random.rand(1:c.num_depots)
    depot = c.depots[d]

    swap_1 = Random.rand(1:length(depot.route_encoding))
    swap_2 = Random.rand(1:length(depot.route_encoding))

    temp = depot.route_encoding[swap_1]
    depot.route_encoding[swap_1] = depot.route_encoding[swap_2]
    depot.route_encoding[swap_2] = temp

    new_depot = Depot(
        depot.route_encoding, 
        1, 
        Dict(),
        Vector(),
        Vector(), 
        problem_params.depot_info[d][3], 
        problem_params.depot_info[d][4],
        problem_params.max_vehicles, 
        d
        )
    route_scheduler!(new_depot, problem_params.customer_info, problem_params.distances, problem_params.num_depots)
    c.depots[d] = new_depot
    calculate_chromosome_fitness!(c)

end


end

