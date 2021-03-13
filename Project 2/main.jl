include("FileParser.jl")
include("MDVRProblem.jl")
using .FileParser: mdvrp_parser
using .MDVRProblem: calculate_distances, nearest_depot

f = open("Testing Data/Data Files/p01", "r")
num_depots, num_customers, depot_info, customer_info = mdvrp_parser(f)
close(f)
distances = calculate_distances(num_customers, num_depots, depot_info, customer_info)
nearest_depot_dict = nearest_depot(distances, num_depots, num_customers)

println(customer_clusters[1])
println(distances[1+num_depots, 1:4])