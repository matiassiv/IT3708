import genetic

test_1 = genetic.Individual("01001")
test_2 = genetic.Individual("11100")

assert genetic.hamming_distance(test_1, test_2) == 3
