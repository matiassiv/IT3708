include("FileParser.jl")
using .FileParser: mdvrp_parser

f = open("Testing Data/Data Files/p01", "r")
depot_info, customer_info = mdvrp_parser(f)
close(f)
println(depot_info)
println()
println(customer_info)