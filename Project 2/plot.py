import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename) as f:
        text = f.readlines()
    return text


def get_coordinates(filename):
    problem_instance = [line.strip() for line in read_file(filename)]

    num_customers = list(map(int, problem_instance[0].split(' ')))[1]
    num_depots = list(map(int, problem_instance[0].split(' ')))[2]

    customer_coord = []
    customer_end_index = num_customers+num_depots+1

    for customer in range(num_depots+1, customer_end_index):
        curr_customer_info = list(map(int, problem_instance[customer].split()))
        customer_coord.append((curr_customer_info[1], curr_customer_info[2]))

    depot_coord = []
    for depot in range(customer_end_index, customer_end_index+num_depots):
        curr_depot_info = list(map(int, problem_instance[depot].split()))
        depot_coord.append((curr_depot_info[1], curr_depot_info[2]))

    return customer_coord, depot_coord


def get_routes(filename):
    content = [line.strip() for line in read_file(filename)]

    solution_cost = float(content[0])
    routes = {}

    for i in range(1, len(content)):
        line = content[i].split()
        depot_id = int(line[0])
        if depot_id not in routes:
            routes[depot_id] = {}
        route_id = int(line[1])
        route = list(map(int, line[5:-1]))
        routes[depot_id][route_id] = route

    return solution_cost, routes


def plot_solution(problem_file, solution_file):

    customer_coord, depot_coord = get_coordinates(problem_file)
    solution_cost, routes = get_routes(solution_file)

    fig, ax = plt.subplots(num=None, figsize=(10, 10), dpi=100)
    fig.canvas.set_window_title('IT3708: Project 2')
    #colormap = plt.cm.Set1
    #plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 10)])

    for i, depot in enumerate(depot_coord):
        curr_depot = depot_coord[i]

        for route_id, route in routes[i+1].items():
            tot_route = [curr_depot]

            for customer in route:
                tot_route.append(customer_coord[customer-1])

            tot_route.append(curr_depot)
            plt.plot(*zip(*tot_route), linewidth=2.0, zorder=-1)

    customers = plt.scatter(*zip(*customer_coord),
                            c='#00ABFF', label='Customers', s=75, zorder=1)
    depots = plt.scatter(*zip(*depot_coord),
                         c='#FFBA00', label='Depots', s=100, zorder=2)

    plt.title("Data set: " + solution_file[:3] +
              " with duration: " + str(solution_cost))
    plt.legend()

    plt.show()


plot_solution("Testing Data/Data Files/p06", "p06.txt")
