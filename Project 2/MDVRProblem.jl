module MDVRProblem
using Random
export init_random_chromosome, Chromosome, Depot

mutable struct Depot
    route_encoding::Vector{Int} # Represents the genotype of depot
    num_routes::Int
    routes::Dict{Int, Vector{Int}}  # Represents the phenotype of depot
    fitness::Float64
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
 

function init_random_chromosome(
    num_depots::Int, 
    num_customers::Int,
    max_vehicles::Int,  
    depot_assignments, 
    depot_info,
    customer_info,
    distances)
    chromosome = Chromosome(0.0, num_depots, num_customers, Vector{typeof(Depot)}(undef, num_depots))
    for i = 1:num_depots
        chromosome.depots[i] = Depot(
            Random.shuffle(depot_assignments[i]), 
            1, 
            Dict(), 
            0.0, 
            depot_info[i][3], 
            depot_info[i][4],
            max_vehicles, 
            i
            )
        # Generate route and check if it is feasible. If not, then generate another chromosome
        while !(route_scheduler!(chromosome.depots[i], customer_info, distances, num_depots))
            chromosome.depots[i] = Depot(
            Random.shuffle(depot_assignments[i]), 
            1, 
            Dict(), 
            0.0, 
            depot_info[i][3], 
            depot_info[i][4],
            max_vehicles, 
            i
            )
        end
        chromosome.fitness += chromosome.depots[i].fitness
    
    end
    return chromosome
end


function assign_customer!(nearest_depot_dict, chromosome::Chromosome)
    for (key, val) in nearest_depot_dict
        push!(chromosome.depots[val].customer_indices, key)
    end
end


# Convergence speed will most likely heavily rely on how effective the
# route scheduler is. Possible to add customized heuristics to improve
# performance, like adding checks for whether starting a new route will 
# improve fitness or not.

function route_scheduler!(depot::Depot, customer_info, distances, num_depots)
    # Customer info is an array of tuples, where index of array == customer_id
    # and the tuples are of the form (xpos, ypos, customer_demand)
    # distances is an array of arrays designating all the interdistances between customers and depots
    # where all customer id's are offset by num_depots

    # Phase 1
    curr_route = 1
    route_duration = [0.0]
    route_load = [0]
    depot.routes[curr_route] = []
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
            push!(depot.routes[curr_route], depot.route_encoding[i]) # Add to route
            route_duration[curr_route] += distances[prev_id, new_id] # Increment route duration
            route_load[curr_route] += demand                         # Increment route load usage
            prev_id = new_id                                         # Set customer as previously visited
            added_to_route = true                                    # Update flag
        
        # If we cannot add to route, we add distance back to depot and start a new route
        else
            route_duration[curr_route] += distances[new_id, depot.id] # Add return to depot distance
            curr_route += 1                                           # Increment number of routes
            depot.routes[curr_route] = []                             # Add new route
            push!(depot.routes[curr_route], depot.route_encoding[i])  # Add customer to new route
            push!(route_duration, distances[depot.id, new_id])        # Add duration of new customer
            push!(route_load, demand)                                 # Add load of new customer
            prev_id = new_id                                          # Set customer as previously visited
        end
        
        # Check if we have reached final customer
        if i == length(depot.route_encoding)
            route_duration[curr_route] += distances[new_id, depot.id] # Add final distance back to depot
            if (
                (depot.max_route_duration != 0 && route_duration[curr_route] > depot.max_route_duration) 
                || curr_route > depot.max_routes
                )
                return false
            end
        end
    end
    
    depot.num_routes = curr_route # Set total number of routes
    # Phase 2 
    # Check if we can optimize route selection by setting last customer
    # of route r to be the first customer in route r+1. Still need to boundary check, though.
    improvement = true
    while improvement
        for i = 1:depot.num_routes-1
            last_customer = depot.routes[i][end]
            next_to_last = depot.routes[i][end-1]
            without_last = depot.routes[i][1:end-1]
            # Find duration of route without last element
            without_last_duration = (
                route_duration[i]
                - distances[last_customer + num_depots][depot.id]     # Subtract last customer->depot
                - distances[next_to_last + num_depots][last_customer] # Subtract next_to_last->last customer
                + distances[next_to_last + num_depots][depot.id]      # Add next_to_last->depot
            )

            added_customer = [last_customer] + depot.routes[i+1][:]

    end
    
    depot.fitness = sum(route_duration)
    return true
end

function valid_duration(depot_id, max_route_duration, distances, prev_id, new_id, curr_route_duration)
    # If max_route_duration == 0, then route can be as long as it wants
    return (max_route_duration == 0 ||
        distances[prev_id, new_id] 
        + curr_route_duration 
        + distances[new_id, depot_id]
        <= max_route_duration
        )
end

function valid_load(max_route_load, demand, curr_route_load) 
    return demand + curr_route_load <= max_route_load
end

function shorter_than_new_route(distances, depot_id::Int, prev_id::Int, new_id::Int)
    return (distances[prev_id, new_id] <= distances[depot_id, new_id])
end

end

