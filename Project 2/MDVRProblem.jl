module MDVRProblem
include("Utils.jl")
using .Utils: ProblemInstance
using Random
export init_random_chromosome, Chromosome, Depot, check_distance

mutable struct Depot
    route_encoding::Vector{Int} # Represents the genotype of depot
    num_routes::Int
    routes::Dict{Int, Vector{Int}}  # Represents the phenotype of depot
    route_loads::Vector{Int}
    route_durations::Vector{Float64}
    max_route_duration::Int
    max_route_load::Int
    max_routes::Int
    id::Int
end

mutable struct Chromosome
    fitness::Float64
    num_depots::Int
    num_customers::Int
    depots::Vector{Depot}
end

 
# Initialization of a chromosome for the original population
function init_random_chromosome(
    num_depots::Int, 
    num_customers::Int,
    max_vehicles::Int,  
    depot_assignments::Dict{Int, Vector{Int}}, 
    depot_info::Vector{NTuple{4, Int64}},
    customer_info::Vector{NTuple{3, Int64}},
    distances::Array{Float64, 2}
    )::Chromosome
    chromosome = Chromosome(0.0, num_depots, num_customers, Vector{typeof(Depot)}(undef, num_depots))
    for i = 1:num_depots
        chromosome.depots[i] = Depot(
            Random.shuffle(depot_assignments[i]), 
            1, 
            Dict(),
            Vector(),
            Vector(), 
            depot_info[i][3], 
            depot_info[i][4],
            max_vehicles, 
            i
            )
        # Generate route and add penalties if not feasible
        chromosome.fitness += (route_scheduler!(chromosome.depots[i], customer_info, distances, num_depots))   
    end
    return chromosome
end

# Deterministic initialization of a chromosome
function init_chromosome(
    num_depots::Int, 
    num_customers::Int,
    max_vehicles::Int,  
    depot_assignments::Dict{Int, Vector{Int}}, 
    depot_info::Vector{NTuple{4, Int64}},
    customer_info::Vector{NTuple{3, Int64}},
    distances::Array{Float64, 2},
    #problem_params::ProblemInstance
    )::Chromosome
    chromosome = Chromosome(0.0, num_depots, num_customers, Vector{typeof(Depot)}(undef, num_depots))
    for i = 1:num_depots
        chromosome.depots[i] = Depot(
            depot_assignments[i], 
            1, 
            Dict(),
            Vector(),
            Vector(), 
            depot_info[i][3], 
            depot_info[i][4],
            max_vehicles, 
            i
            )
        # Generate route and add penalties if it's not feasible. 
        chromosome.fitness += (route_scheduler!(chromosome.depots[i], customer_info, distances, num_depots))
    
    end
    return chromosome
end

#=
function assign_customer!(nearest_depot_dict, chromosome::Chromosome)
    for (key, val) in nearest_depot_dict
        push!(chromosome.depots[val].customer_indices, key)
    end
end
=#

# Convergence speed will most likely heavily rely on how effective the
# route scheduler is. Possible to add customized heuristics to improve
# performance, like adding checks for whether starting a new route will 
# improve fitness or not.

function route_scheduler!(
    depot::Depot, 
    customer_info::Vector{NTuple{3, Int64}}, 
    distances::Array{Float64, 2}, 
    num_depots::Int
    )::Float64
    # Customer info is an array of tuples, where index of array == customer_id
    # and the tuples are of the form (xpos, ypos, customer_demand)
    # distances is an array of arrays designating all the interdistances between customers and depots
    # where all customer id's are offset by num_depots

    # Phase 1
    routes = Dict{Int, Vector{Int}}()
    curr_route = 1
    route_duration = [0.0]
    route_load = [0]
    routes[curr_route] = []
    prev_id = depot.id

    for i = 1:length(depot.route_encoding)
        new_id = depot.route_encoding[i] + num_depots
        demand = customer_info[depot.route_encoding[i]][3]
        # Check if customer can be added to current route
        if (
            valid_duration(depot.id, depot.max_route_duration, distances, prev_id, new_id, route_duration[curr_route])
            && valid_load(depot.max_route_load, demand, route_load[curr_route])
            #&& shorter_than_new_route(distances, depot.id, prev_id, new_id)
        )
            push!(routes[curr_route], depot.route_encoding[i]) # Add to route
            route_duration[curr_route] += distances[prev_id, new_id] # Increment route duration
            route_load[curr_route] += demand                         # Increment route load usage
            prev_id = new_id                                         # Set customer as previously visited
        
        # If we cannot add to route, we add distance back to depot and start a new route
        else
            route_duration[curr_route] += distances[prev_id, depot.id] # Add return to depot distance
            curr_route += 1                                            # Increment number of routes
            routes[curr_route] = []                              # Add new route
            push!(routes[curr_route], depot.route_encoding[i])   # Add customer to new route
            push!(route_duration, distances[depot.id, new_id])         # Add duration of new customer
            push!(route_load, demand)                                  # Add load of new customer
            prev_id = new_id                                           # Set customer as previously visited
        end
        #@assert curr_route == length(route_duration)
        
        # Check if we have reached final customer
        if i == length(depot.route_encoding)
            route_duration[curr_route] += distances[new_id, depot.id] # Add final distance back to depot
        end
    end


    # Phase 2 
    # Check if we can optimize route selection by setting last customer
    # of route r to be the first customer in route r+1, while still maintaining
    # the constraints imposed by the problem. The optimization is based on trying
    # to reduce overall duration of all routes.
    improvement = true
    counter = 0
    while improvement
        counter += 1
        improvement = false
        for i = depot.num_routes-1:-1:1
            # Check that length of route is longer than 1
            if length(routes[i]) < 2
                continue
            end
            last_customer = routes[i][end]
            last_customer_demand = customer_info[last_customer][3]
            next_to_last = routes[i][end-1]
            without_last = copy(routes[i][1:end-1])
            # Find duration of route without last element
            without_last_duration = (
                route_duration[i]
                - distances[last_customer + num_depots, depot.id]     # Subtract last customer->depot
                - distances[next_to_last + num_depots, last_customer+num_depots] # Subtract next_to_last->last customer
                + distances[next_to_last + num_depots, depot.id]      # Add next_to_last->depot
            )
            @assert abs(check_distance(depot.id, num_depots, without_last, distances) - without_last_duration) < 0.001

            # Add last element of r to r+1
            added_customer = copy(routes[i+1])
            pushfirst!(added_customer, last_customer)
            new_second = routes[i+1][1]
            added_duration = (
                route_duration[i+1]
                - distances[depot.id, new_second+num_depots]                 # Subtract depot->second customer
                + distances[depot.id, last_customer+num_depots]              # Add depot->new first customer
                + distances[last_customer+num_depots, new_second+num_depots] # Add new first-> new second
            )
            @assert abs(check_distance(depot.id, num_depots, added_customer, distances) - added_duration) < 0.001

            # Check if new route has valid load and duration
            if (
                route_load[i+1] + last_customer_demand <= depot.max_route_load                   # Check valid load
                && (depot.max_route_duration == 0 || added_duration <= depot.max_route_duration) # Check valid duration
                && without_last_duration+added_duration < route_duration[i]+route_duration[i+1]  # Check smaller duration
            )
                # Update route r+1
                routes[i+1] = added_customer
                route_duration[i+1] = added_duration
                route_load[i+1] += last_customer_demand
            
                # Update route r
                routes[i] = without_last
                route_duration[i] = without_last_duration
                route_load[i] -= last_customer_demand
                
                # Set flag variable so we can run another check
                improvement = true
            end
            #@assert length(routes[i]) > 0
            #@assert length(routes[i+1]) > 0
        end

    end
    for (key, val) in routes
        @assert abs(check_distance(depot.id, num_depots, val, distances) - route_duration[key]) < 0.001
    end
    
    depot.num_routes = length(route_duration)
    fitness = sum(route_duration)
    if depot.num_routes > depot.max_routes
        # Add worse penalty depending on how many more routes than max_routes there are
        pen = depot.num_routes - depot.max_routes
        fitness += pen * 200
    end

    depot.route_durations = route_duration
    depot.route_loads = route_load
    depot.routes = routes
    return fitness
end

function valid_duration(
    depot_id::Int, 
    max_route_duration::Int, 
    distances::Array{Float64, 2}, 
    prev_id::Int, 
    new_id::Int, 
    curr_route_duration::Float64
    )
    # If max_route_duration == 0, then route can be as long as it wants
    return (max_route_duration == 0 ||
        distances[prev_id, new_id] 
        + curr_route_duration 
        + distances[new_id, depot_id]
        <= max_route_duration
        )
end

function valid_load(max_route_load::Int, demand::Int, curr_route_load::Int) 
    return demand + curr_route_load <= max_route_load
end
#=
function shorter_than_new_route(distances, depot_id::Int, prev_id::Int, new_id::Int)
    return (distances[prev_id, new_id] <= distances[depot_id, new_id])
end
=#

function check_distance(depot_id::Int, num_depots::Int, route::Vector{Int}, distances::Array{Float64, 2})
    distance = 0.0
    #Add first distance
    distance += distances[depot_id, route[1]+num_depots]
    for i = 1:length(route)-1
        distance += distances[num_depots+route[i], num_depots+route[i+1]]
    end
    distance += distances[num_depots+route[end], depot_id]
    return distance
end

function check_load(route::Vector{Int}, customer_info::Vector{NTuple{3, Int64}})
    demand = 0
    for customer in route
        demand += customer_info[customer][3]
    end
    return demand
end

end


