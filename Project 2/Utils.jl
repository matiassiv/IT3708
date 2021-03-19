module Utils
export calculate_distances, nearest_depot, ProblemInstance

struct ProblemInstance
    num_depots::Int
    num_customers::Int
    max_vehicles::Int
    depot_info::Vector{NTuple{4, Int64}}
    customer_info::Vector{NTuple{3, Int64}}
    distances::Array{Float64, 2}
    borderline_customers::Dict{Int, Set{Int}}
    depot_assignments::Dict{Int, Vector{Int}}
end

function calculate_euclidean(x1, y1, x2, y2)
    return sqrt((x1-x2)^2 + (y1-y2)^2)
end

function calculate_distances(
    num_customers::Int,
    num_depots::Int,
    depots::AbstractVector{NTuple{4, Int64}},
    customers::AbstractVector{NTuple{3, Int64}}
    )
    # Calculates all interdistances

    distances = zeros((num_customers+num_depots, num_customers+num_depots))

    for i = 1:num_depots
        x1, y1 = depots[i][1], depots[i][2]

        # Calculate inter-depot distances
        for j = 1:num_depots
            x2, y2 = depots[j][1], depots[j][2]
            distances[i, j] = calculate_euclidean(x1,y1,x2,y2)       
        end

        # Calculate depot-customer distances
        for j = 1:num_customers
            x2, y2 = customers[j][1], customers[j][2]
            distances[i, j+num_depots] = calculate_euclidean(x1,y1,x2,y2)
        end
    end

    for i = 1:num_customers
        x1, y1 = customers[i][1], customers[i][2]
        # Calculate customer-depot distances
        for j = 1:num_depots
            x2, y2 = depots[j][1], depots[j][2]
            distances[num_depots+i, j] = calculate_euclidean(x1,y1,x2,y2)
        end

        # Calculate inter-customer distances
        for j = 1:num_customers
            x2, y2 = customers[j][1], customers[j][2]
            distances[num_depots+i, num_depots+j] = calculate_euclidean(x1,y1,x2,y2)
        end
    end
    return distances
end

# Calculates the nearest depot for each customer
function nearest_depot(distances::Array{Float64, 2}, num_depots::Int, num_customers::Int, depot_info, bound::Float64=0.1)
    # Returns two dictionaries
    #   nearest_depot_dict = Dict(customer_id => nearest_depot_id)
    #   depot_assignments = Dict(depot_id => [customer_ids])

    nearest_depot_dict = Dict{Int, Vector{Float64}}()

    # TODO Might need to implement borderline check for depot reassignments
    #borderline_customers = Dict()

    for i = num_depots+1 : num_depots+num_customers
        nearest_depot_dict[i-num_depots] = Vector(undef, num_depots)
        for j = 1:num_depots
            nearest_depot_dict[i-num_depots][j] = distances[i, j]
        end
    end

    # Find borderline customers
    borderline_customers = Dict{Int, Set{Int}}()
    for (key, val) in nearest_depot_dict
        borderline_depots = []
        sorted = sort(val)
        for i = 2:num_depots
            if (sorted[i]-sorted[1]) / sorted[1] < bound && 2 * sorted[i] <= depot_info[1][3]
                append!(borderline_depots, findall(x->x==sorted[i], nearest_depot_dict[key]))
            end
        end
        
        if length(borderline_depots) > 0
            append!(borderline_depots, findall(x->x==sorted[1], nearest_depot_dict[key]))
            borderline_customers[key] = Set(borderline_depots)
        end
        
    end


    # Assign to initial closest depot
    depot_assignments = Dict{Int, Vector{Int}}()
    for i = 1:num_depots
        depot_assignments[i] = []
    end
    for (key, val) in nearest_depot_dict
        push!(depot_assignments[argmin(val)], key)
    end

    return borderline_customers, depot_assignments
end




end

