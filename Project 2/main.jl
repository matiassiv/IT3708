include("FileParser.jl")
include("Utils.jl")
include("MDVRProblem.jl")
include("Parameters.jl")
include("GeneticAlgorithm.jl")
using .FileParser: mdvrp_parser
using .Utils: calculate_distances, nearest_depot, ProblemInstance
using .MDVRProblem: Depot, Chromosome, init_random_chromosome, check_distance
using .Parameters: Params
using .GeneticAlgorithm: GA

function main()
    # Load relevant params
    p = Params()

    # Open problem file
    f = open(p.PROBLEM_FILEPATH, "r")
    num_depots, num_customers, max_vehicles, depot_info, customer_info = mdvrp_parser(f)
    close(f)
    println("Num_depots: ", num_depots)
    println("Num customers: ", num_customers)
    println("Max vehicles: ", max_vehicles)
    #=
    for i = 1:length(depot_info)
        println(i, ": ", depot_info[i])
    end
    for i = 1:length(customer_info)
        println(i, ": ", customer_info[i])
    end
    =#
    # Calculate relevant interdistances and find nearest depots
    distances = calculate_distances(num_customers, num_depots, depot_info, customer_info)
    borderline_customers, depot_assignments = nearest_depot(distances, num_depots, num_customers, depot_info)
    
    for (key, val) in depot_assignments
        println(key, ": ", val)
    end
    
    println(borderline_customers)

    problem_params = ProblemInstance(
        num_depots, 
        num_customers, 
        max_vehicles,
        depot_info,
        customer_info,
        distances,
        borderline_customers,
        depot_assignments 
        )

    best = GA(350, 250, problem_params)

    println("BEST FITNESS: ", best.fitness)
    total_customers = []
    for i = 1:length(best.depots)
        println()
        println("SET DIFFERENCE FROM ORIGINAL ", setdiff(depot_assignments[i], best.depots[i].route_encoding))
        @assert num_routes <= best.depots[i].max_routes

        num_routes = best.depots[i].num_routes
        for j = 1:num_routes
            route = best.depots[i].routes[j]
            @assert check_distance(i, num_depots, route, distances) == best.depots[i].route_durations[j]
            @assert best.depots[i].route_loads[j] <= depot_info[i][4]

            append!(total_customers, best.depots[i].routes[j])
            println(i, " ", j, " ", best.depots[i].route_durations[j], " ", best.depots[i].route_loads[j], " ", best.depots[i].routes[j])
        end
    end
    println("NUMBER OF CUSTOMERS: ", num_customers, " NUM SERVED: ", length(unique(total_customers)))
end
