import networkx as net
from scipy import spatial


def example_graph_paper():
    G = net.Graph()

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
    G.add_edge('u', 'u_2', type='f')

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


def sim(G, u1, u2):
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


def augment(G, beta):
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
            similarity = sim(G, u, x)
            sum += similarity

        for v in users_setminus_u:
            u_locations = set(visited_locations_by_user(G, u))
            v_locations = set(visited_locations_by_user(G, v))
            have_common_location_visited = (0 < len(u_locations & v_locations))

            are_friends = G.has_edge(u, v)

            if have_common_location_visited and are_friends:
                probability = ((1 - beta) * 1 / F_u) + ((beta * sim(G, u, v)) / sum)
            elif have_common_location_visited:
                probability = (beta * sim(G, u, v)) / sum
            elif are_friends:
                probability = (1 - beta) * 1 / F_u
            else:
                continue
            G_augmented.add_edge(u, v, weight=probability)

    print(f"Augmented graph: {G_augmented.edges(data=True)}")
    return G_augmented


def LFBCA(G, u, d, beta, N, location_friendship):
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
        G_augmented = augment(G, beta)
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
            s[locations.index(l)] = s[locations.index(l)] + (pi[U_setminus_u.index(v)] * w_v_l)

    # 7
    R = []
    for l in L_setminus_L_u:
        # Get value of min element in R
        R.sort(key=lambda x: x[1])
        if 0 < len(R):
            min_element_value = R[0][1]

        # Ignore the geodist requirement, should be implemented here
        if len(R) < N:
            R.append((l, s[locations.index(l)]))
        elif s[locations.index(l)] <= min_element_value:
            continue
        else:
            R.pop(0)
            R.append((l, s[locations.index(l)]))

    R.sort(key=lambda x: -x[1])
    return R


def main():
    G = example_graph_paper()
    u = 'u_2'
    d = 2.0
    beta = 0.8
    N = 2

    recommendations = LFBCA(G, u, d, beta, N, True)
    print(f"recommendations: {recommendations}")


if __name__ == "__main__":
    main()
