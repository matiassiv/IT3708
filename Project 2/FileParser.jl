module FileParser

export mdvrp_parser

function mdvrp_parser(f)
    # Parser for the textfiles describing MDVRP problems
    # Takes in filename as a String
    # Returns two ints and two Vectors of tuples, where index refers to the particular customer/depot
    #   num_depots = total number of depots in problem
    #   num_customers = total number of customers in problem
    #   depot_info = [(xpos, ypos, max_duration, max_load)]
    #   customer_info = [(xpos, ypos, demand)]
    a = readlines(f)

    line1 = split(a[1])
    max_vehicles, num_customers, num_depots = parse(Int, line1[1]), parse(Int, line1[2]), parse(Int, line1[3])
    depot_info = Vector{NTuple{4,Int64}}(undef, num_depots)

    for i = 2 : 2+num_depots-1
        # Get constraints for each depot
        depot_constraints = split(a[i])
        depot_max_duration, vehicle_max_load = parse(Int, depot_constraints[1]), parse(Int, depot_constraints[2])

        # Get position for each depot
        i_depot = i + num_customers + num_depots
        depot_pos = split(a[i + num_customers + num_depots])
        depot_xpos, depot_ypos = parse(Int, depot_pos[2]), parse(Int, depot_pos[3])

        depot_info[i - 1] = (depot_xpos, depot_ypos, depot_max_duration, vehicle_max_load)
    end

    customer_info = Vector{NTuple{3,Int64}}(undef, num_customers)

    for i = 2+num_depots : 2+num_depots+num_customers-1
        # Get the required info for each customer
        cus = split(a[i])
        cus_xpos, cus_ypos, cus_demand = parse(Int, cus[2]), parse(Int, cus[3]), parse(Int, cus[5])

        customer_info[i - num_depots - 1] = (cus_xpos, cus_ypos, cus_demand)
    end
    close(f)

    return num_depots, num_customers, max_vehicles, depot_info, customer_info

end

end
