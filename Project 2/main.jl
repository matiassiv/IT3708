include("FileParser.jl")
include("Utils.jl")
include("MDVRProblem.jl")
include("Parameters.jl")
using .FileParser: mdvrp_parser
using .Utils: calculate_distances, nearest_depot
using .MDVRProblem: Depot, Chromosome, init_random_chromosome
using .Parameters: Params

# Load relevant params
p = Params()

# Open problem file
f = open(p.PROBLEM_FILEPATH, "r")
num_depots, num_customers, max_vehicles, depot_info, customer_info = mdvrp_parser(f)
close(f)

# Calculate relevant interdistances and find nearest depots
distances = calculate_distances(num_customers, num_depots, depot_info, customer_info)
nearest_depot_dict, depot_assignments = nearest_depot(distances, num_depots, num_customers)

# Initialize three chromosomes with random route_encodings
function test_viability()
    for j = 1:400
        c = init_random_chromosome(num_depots, num_customers, max_vehicles, depot_assignments, depot_info, customer_info, distances)
        #c2 = init_random_chromosome(num_depots, num_customers, depot_assignments, depot_info, customer_info, distances)
        #c3 = init_random_chromosome(num_depots, num_customers, depot_assignments, depot_info, customer_info, distances)

        #println("###################################################")
        #println("Customers placed at depot:")
        #for i = 1:num_depots
        #    println(i, " ", c.depots[i].route_encoding, " Load: ", c.depots[i].max_route_load, " Duration: ", c.depots[i].max_route_duration)
        #    println("Routes: ", c.depots[i].routes, " fitness: ", sum(c.depots[i].route_durations))
        #end
        #println(c.fitness)
    end
end

test_viability()
"""
for i = 1:num_depots
    println(i, " ", c2.depots[i].route_encoding, " Load: ", c2.depots[i].max_route_load, " Duration: ", c2.depots[i].max_route_duration)
end
for i = 1:num_depots
    println(i, " ", c3.depots[i].route_encoding, " Load: ", c3.depots[i].max_route_load, " Duration: ", c3.depots[i].max_route_duration)
end
"""


