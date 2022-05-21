from pprint import pprint

import networkx as net
from scipy import spatial


def prof_stud_graph():
    G = net.DiGraph()

    G.add_node('Univ')
    G.add_node('ProfA')
    G.add_node('StudentA')
    G.add_node('ProfB')
    G.add_node('StudentB')

    G.add_edge('Univ', 'ProfA')
    G.add_edge('Univ', 'ProfB')
    G.add_edge('ProfA', 'StudentA')
    G.add_edge('StudentA', 'Univ')
    G.add_edge('ProfB', 'StudentB')
    G.add_edge('StudentB', 'ProfB')

    return G


def baking_shopping_graph():
    G = net.DiGraph()

    # Shoppers
    G.add_node('A', type='shopper')
    G.add_node('B', type='shopper')

    # Ingredients
    G.add_node('sugar', type='ingredient')
    G.add_node('frosting', type='ingredient')
    G.add_node('eggs', type='ingredient')
    G.add_node('flour', type='ingredient')

    G.add_edge('A', 'sugar')
    G.add_edge('A', 'frosting')
    G.add_edge('A', 'eggs')
    G.add_edge('B', 'frosting')
    G.add_edge('B', 'eggs')
    G.add_edge('B', 'flour')

    return G


def example_graph_paper():
    G = net.DiGraph()

    # Locations
    G.add_node('l_1', type='location')
    G.add_node('l_2', type='location')
    G.add_node('l_3', type='location')
    G.add_node('l_4', type='location')
    G.add_node('l_5', type='location')

    # Users
    G.add_node('u', type='user')
    G.add_node('u_1', type='user')
    G.add_node('u_2', type='user')
    G.add_node('u_3', type='user')

    # Visit edges
    G.add_edge('u_1', 'l_1', weight=7, type='v')
    G.add_edge('u_1', 'l_2', weight=3, type='v')
    G.add_edge('u', 'l_2', weight=1, type='v')
    G.add_edge('u', 'l_3', weight=1, type='v')
    G.add_edge('u', 'l_4', weight=8, type='v')
    G.add_edge('u_3', 'l_4', weight=6, type='v')
    G.add_edge('u_3', 'l_5', weight=4, type='v')

    # Friend edges
    G.add_edge('u_1', 'u', type='f')
    G.add_edge('u', 'u_1', type='f')
    G.add_edge('u', 'u_2', type='f')
    G.add_edge('u_2', 'u_2', type='f')

    return G


def get_nodes_by_type(G, type):
    return list((n for n in G if G.nodes[n]['type'] == type))


def get_edges_by_type(G, type):
    return list(n[0] for n in list(G.edges(data=True)) if n[2]['type'] == type)


def visited_locations_by_user(G, u):
    location_nodes = get_nodes_by_type(G, 'location')
    locations = []
    for node in list(G.neighbors(u)):
        if node in location_nodes:
            locations.append(node)
    return locations


def friends_of_user(G, u):
    user_nodes = get_nodes_by_type(G, 'user')
    users = []
    for node in list(G.neighbors(u)):
        if node in user_nodes:
            users.append(node)
    return users


def PPR_values_Pi_v_friends_importance(G, u, alpha, epsilon):
    U = get_nodes_by_type(G, 'user')

    pi = [0] * len(U)

    # Get number of friends list
    d = [0] * len(U)
    for index, i in enumerate(U):
        d[index] = len(friends_of_user(G, i))

    b = [0] * len(U)
    b[U.index(u)] = 1

    while True:
        b_before = b
        for i in U:
            if b[U.index(i)] < epsilon:
                continue
            pi[U.index(i)] = pi[U.index(i)] + (1 - alpha) * b[U.index(i)]
            for friend_j in friends_of_user(G, i):
                b[U.index(friend_j)] = b[U.index(friend_j)] + alpha * b[U.index(i)] / d[U.index(i)]
        if b_before == b:
            break
    return pi


def simrank_not_mine(G, u, v):
    return net.simrank_similarity(G, u, v, importance_factor=0.8)


def equation_1(a, b, in_neighbours_a, c, R, G):
    if a == b:
        return 1
    # Find the I(b)
    in_neighbours_b = []
    for edge in G.in_edges(b):
        in_neighbours_b.append(edge[0])

    if (len(in_neighbours_a) == 0 or len(in_neighbours_b) == 0):
        return 0
    # These are the two sums
    sum = 0
    for in_neighbour_a in in_neighbours_a:
        for in_neighbour_b in in_neighbours_b:
            sum += (R.get(in_neighbour_a)).get(in_neighbour_b)

    # Put it together and find new value

    fraction = c / (G.in_degree(a) * G.in_degree(b))

    return fraction * sum


def apply_iteration_on_R_naive(G, R, c):
    # Apply iterations from equation 5 on page 5 in the paper
    R_1 = {}

    for a, a_value in R.items():
        a_dict = {}

        # Find the I(a)
        in_neighbours_a = []
        for edge in G.in_edges(a):
            in_neighbours_a.append(edge[0])

        for b, b_value in a_value.items():
            a_dict[b] = equation_1(a, b, in_neighbours_a, c, R, G)
        R_1[a] = a_dict
    return R_1


def equation_2(a, b, out_neighbours_a, c_1, R, G):
    if a == b:
        return 1
    # Find the O(b)
    out_neighbours_b = []
    for edge in G.out_edges(b):
        out_neighbours_b.append(edge[0])

    if len(out_neighbours_a) == 0 or len(out_neighbours_b) == 0:
        return 0
    # These are the two sums
    sum = 0
    for out_neighbour_a in out_neighbours_a:
        for out_neighbour_b in out_neighbours_b:
            sum += (R.get(out_neighbour_a)).get(out_neighbour_b)

    # Put it together and find new value

    fraction = c_1 / (G.out_degree(a) * G.out_degree(b))

    return fraction * sum


def equation_3(c, d, in_neighbours_c, c_2, R, G):
    if c == d:
        return 1
    # Find the O(b)
    in_neighbours_d = []
    for edge in G.in_edges(d):
        in_neighbours_d.append(edge[0])

    if len(in_neighbours_c) == 0 or len(in_neighbours_d) == 0:
        return 0
    # These are the two sums
    sum = 0
    for in_neighbour_c in in_neighbours_c:
        for in_neighbour_d in in_neighbours_d:
            sum += (R.get(in_neighbour_c)).get(in_neighbour_d)

    # Put it together and find new value
    fraction = c_2 / (G.in_degree(c) * G.in_degree(d))
    return fraction * sum


def apply_iteration_on_R_bipartite(G, R, c, out_type, in_type):
    all_out_type_nodes = get_nodes_by_type(G, out_type)
    all_in_type_nodes = get_nodes_by_type(G, in_type)
    # Todo!
    minmax = True

    R_1 = {}

    for a, a_value in R.items():
        a_dict = {}
        for b, b_value in a_value.items():
            if a in all_out_type_nodes and b in all_out_type_nodes:
                # Find the O(a)
                out_neighbours_a = []
                for edge in G.out_edges(a):
                    out_neighbours_a.append(edge[0])

                a_dict[b] = equation_2(a, b, out_neighbours_a, c, R, G)

            elif a in all_in_type_nodes and b in all_in_type_nodes:
                # Find the I(a)
                in_neighbours_a = []
                for edge in G.in_edges(a):
                    in_neighbours_a.append(edge[0])
                a_dict[b] = equation_3(a, b, in_neighbours_a, c, R, G)
            else:
                a_dict[b] = 0
        R_1[a] = a_dict
    return R_1


def apply_iteration_on_R_homogeneous(R, c):
    return {}


def simrank(G, u, v, graph_type):
    c = 0.8
    max_iterations = 100
    minimax = False

    # Initialize R_0
    R_0 = {}
    nodes = G.nodes
    for node in nodes:
        node_dict = {}
        for node2 in nodes:
            if node == node2:
                node_dict[node2] = 1
            else:
                node_dict[node2] = 0
        R_0[node] = node_dict

    iteration_results = [{}] * max_iterations
    iteration_results[0] = R_0
    for k in range(max_iterations - 1):
        if graph_type == "naive":
            iteration_results[k + 1] = apply_iteration_on_R_naive(G, iteration_results[k], c)
        elif graph_type == "bipartite":
            iteration_results[k + 1] = apply_iteration_on_R_bipartite(G, iteration_results[k], c, "shopper",
                                                                      "ingredient")
        elif graph_type == "homogeneous":
            iteration_results[k + 1] = apply_iteration_on_R_homogeneous(iteration_results[k], c)

    return (iteration_results[max_iterations - 1].get(u)).get(v), iteration_results[max_iterations - 1]


def cosine_sim(G, u1, u2):
    def user_profile_p_u(G, u):
        visit_edges = get_edges_by_type(G, 'v')

        profile = [0] * len(visit_edges)
        locations_visited = visited_locations_by_user(G, u)
        sum = 0
        for location in locations_visited:
            sum += G[location][u]['weight']
        for location in locations_visited:
            weight = G[location][u]['weight']
            profile[visit_edges.index(location)] = weight / sum
        return profile

    u1_profile = user_profile_p_u(G, u1)
    u2_profile = user_profile_p_u(G, u2)

    similarity = 1 - spatial.distance.cosine(u1_profile, u2_profile)
    if similarity > 0:
        return similarity
    elif u1_profile == u2_profile:
        return 1
    else:
        return 0


def augment(G, u, beta, similarity_algorithm):
    G_augmented = net.DiGraph()

    # add the user nodes with types
    user_nodes = get_nodes_by_type(G, 'user')
    for user in user_nodes:
        G_augmented.add_node(user, type='user')
    for u in user_nodes:
        F_u = len(friends_of_user(G, u))
        users_setminus_u = get_nodes_by_type(G, 'user')
        users_setminus_u.remove(u)

        sum = 0
        for x in users_setminus_u:
            if similarity_algorithm == "cosine":
                similarity = cosine_sim(G, u, x)
                print(f"Cosine similarity: ({u},{x}): {similarity}")
            elif similarity_algorithm == "simrank":
                similarity = simrank(G, u, x, graph_type="naive")[0]
                print(f"Simrank similarity: ({u},{x}): {similarity}")
            sum += similarity
        for v in users_setminus_u:
            u_locations = set(visited_locations_by_user(G, u))
            v_locations = set(visited_locations_by_user(G, v))
            has_similarity_edge = (0 < len(u_locations & v_locations))
            are_friends = G.has_edge(u, v)

            if has_similarity_edge and are_friends:
                if similarity_algorithm == "cosine":
                    similarity = cosine_sim(G, u, v)
                elif similarity_algorithm == "simrank":
                    similarity = simrank(G, u, v, graph_type="bipartite")[0]
                if sum != 0:
                    probability = ((1 - beta) * 1 / F_u) + ((beta * similarity) / sum)
                else:
                    probability = 0
            elif has_similarity_edge:
                if similarity_algorithm == "cosine":
                    similarity = cosine_sim(G, u, v)
                elif similarity_algorithm == "simrank":
                    similarity = simrank(G, u, v, graph_type="bipartite")[0]
                if sum != 0:
                    probability = (beta * similarity) / sum
                else:
                    probability = 0
            elif are_friends:

                probability = (1 - beta) * 1 / F_u
            else:
                continue
            G_augmented.add_edge(u, v, weight=probability)

    print(f"Augmented graph edges using {similarity_algorithm}: ")
    for edge in G_augmented.edges(data=True):
        print(edge)
    return G_augmented


def LFBCA(G, u, d, beta, N, location_friendship, similarity_algorithm):
    U_setminus_u = get_nodes_by_type(G, 'user')
    U_setminus_u.remove(u)

    # 1
    L_u = visited_locations_by_user(G, u)
    L_setminus_L_u = get_nodes_by_type(G, 'location')
    for location in L_u:
        L_setminus_L_u.remove(location)

    # 2
    alpha = 0.85
    epsilon = 0.01

    if location_friendship:
        G_augmented = augment(G, u, beta, similarity_algorithm)
        pi = PPR_values_Pi_v_friends_importance(G_augmented, u, alpha, epsilon)
    else:
        pi = PPR_values_Pi_v_friends_importance(G, u, alpha, epsilon)

    # 3
    locations = get_nodes_by_type(G, 'location')
    s = [0] * len(locations)

    # 4
    for v in U_setminus_u:
        for l in visited_locations_by_user(G, v):
            w_v_l = G[v][l]["weight"]
            s[locations.index(l)] = s[locations.index(l)] + pi[U_setminus_u.index(v)] * w_v_l

    # 7
    R = []
    for l in L_setminus_L_u:
        R.sort(key=lambda x: x[1])

        if 0 < len(R):
            min_element = R[0][1]

        # Ignore the geodist requirement, should be implemented here
        if len(R) < N:
            R.append((l, s[locations.index(l)]))
        elif s[locations.index(l)] <= min_element:
            continue
        else:
            R.pop(0)
            R.append((l, s[locations.index(l)]))
    return R


def main():
    G = example_graph_paper()
    u = 'u_2'
    d = 2.0
    beta = 0.8
    N = 1

    recommendations = LFBCA(G, u, d, beta, N, location_friendship=True, similarity_algorithm="simrank")
    print(f"recommendation(s): for user: {u} are: {recommendations}")


if __name__ == "__main__":
    main()

    ## Simrank
    # G = prof_stud_graph()
    # single_result, all_results = simrank(G, "Univ", "ProfB", graph_type="naive")
    # pprint(all_results)

    #
    # G = baking_shopping_graph()
    # single_result, all_results = simrank(G, "A", "sugar", graph_type="bipartite")
    # pprint(all_results)
