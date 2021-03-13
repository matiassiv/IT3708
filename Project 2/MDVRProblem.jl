module MDVRProblem
export calculate_distances, nearest_depot

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
    println(size(distances))

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
function nearest_depot(distances::Array{Float64, 2}, num_depots::Int, num_customers::Int)
    nearest_depot_dict = Dict()

    # TODO Might need to implement borderline check for depot reassignments
    #borderline_customers = Dict()

    for i = num_depots+1 : num_depots+num_customers
        min_distance = distances[i, 1]
        depot = 1
        for j = 2 : num_depots
            if distances[i, j] < min_distance
                min_distance = distances[i, j]
                depot = j
            end
        end
        nearest_depot_dict[i-num_depots] = depot
    end
    return nearest_depot_dict
end

end

