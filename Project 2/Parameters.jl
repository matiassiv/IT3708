module Parameters
export Params

struct Params
    FILEPATH::String
    PROBLEM::String
    
    Params(; 
    FILEPATH::String="Testing Data/Data Files/",
    PROBLEM::String="p06"
    ) = new(FILEPATH, PROBLEM)
end
end