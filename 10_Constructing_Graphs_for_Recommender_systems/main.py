import csv as csv
from typing import List, Any
import networkx as net


def get_nodes_by_type(G, type):
    nodes = []
    for node in G.nodes(data='type'):
        if node[1] == type:
            nodes.append(node[0])
    return nodes


# ('rating', 'movie') and also experiment with ((user, time), movie) as in the paper
def NE_Aggregate_user_time__item(G):
    G_prime = net.Graph()

    # Get all the different nodes
    user_nodes = get_nodes_by_type(G, "user")
    time_nodes = get_nodes_by_type(G, "time")
    movie_nodes = get_nodes_by_type(G, "movie")
    rating_nodes = get_nodes_by_type(G, "rating")
    review_nodes = get_nodes_by_type(G, "review")

    for review in review_nodes:
        user = [n for n in G.neighbors(review) if n in user_nodes][0]
        timestamp = [n for n in G.neighbors(review) if n in time_nodes][0]
        rating = [n for n in G.neighbors(review) if n in rating_nodes][0]
        movie = [n for n in G.neighbors(review) if n in movie_nodes][0]

        user_time = float(int(user) + int(timestamp) / 100000000)
        G_prime.add_node(user_time, type='user-time')
        if G_prime.has_edge(user_time, movie):
            # we added this one before, just increase the weight by one
            G_prime[user_time][movie]['weight'] = str(int(G[user_time][movie]['weight']) + 1)
        else:
            # new edge. add with weight=1
            G_prime.add_edge(user_time, movie, weight=1)

        G_prime.add_node(rating, type='rating')
        if G_prime.has_edge(movie, rating):
            # we added this one before, just increase the weight by one
            previous_weight = int(G_prime[movie][rating]['weight'])
            G_prime[movie][rating]['weight'] = str(previous_weight + int(rating))
        else:
            # new edge. add with weight=1
            G_prime.add_edge(movie, rating, weight=rating)
    return G_prime


def G_Merge(graphs):
    if len(graphs) == 1:
        return graphs[0]
    else:
        return net.disjoint_union(graphs[0], G_Merge(graphs[1::]))


def algorithm():
    print("todo")


def normalize(G):
    for node in G.nodes:
        sum = 0
        edges = G.edges(node, data=True)
        for edge in edges:
            weight = edge[2]['weight']
            sum += int(weight)
        for neighbour in G.neighbors(node):
            current_weight_to_neighbour = G[node][neighbour]['weight']
            if sum != 0:
                new_weight = int(current_weight_to_neighbour) / sum
            else:
                new_weight = 0
            G[node][neighbour]['weight'] = new_weight
    return G


def adjust_weights(G, W_tf, W_cf):
    for edge in G.edges(data=True):
        u, v = edge[0], edge[1]
        # contextual -- ie user-time
        if int(u) < 580000:
            factor_weight = W_cf
        # content -- ie rating
        elif 5000000 < int(u):
            factor_weight = W_tf
        else:
            print("shits fucked")
            factor_weight = 1

        current_weight = edge[2].get('weight')
        new_weight = int(current_weight) * factor_weight

        G.add_edge(u, v, weight=new_weight)
    return G


def main():
    file = open('ml-100k/u1.base', 'r')
    moviereader = csv.reader(file, delimiter='\t')

    G = net.Graph()
    review_number = -100000
    user_id_offset = 500000
    movie_id_offset = 5000000
    W_cf = 0.3
    W_tf = 0.7
    for row in moviereader:
        # according to readme the format of the row is userid, movieid, rating, timestamp
        # creating graph with nodes for each item, edges between user and item are through rating
        # (substituting play from the paper) and timestamp is connected to play

        # Review node -- event node type
        G.add_node(review_number, type='review')

        # userid
        G.add_node(str(int(row[0]) + user_id_offset), type='user')

        # itemid
        G.add_node(str(int(row[1]) + movie_id_offset), type='movie')

        # rating
        G.add_node(row[2], type='rating')

        # timestamp
        G.add_node(row[3], type='time')

        # review - movie
        G.add_edge(review_number, str(int(row[0]) + user_id_offset), weight=1)
        # review - rating
        G.add_edge(review_number, str(int(row[1]) + movie_id_offset), weight=1)
        # review - timestamp
        G.add_edge(review_number, row[2], weight=1)
        # review - timestamp
        G.add_edge(review_number, row[3], weight=1)

        review_number -= 1

    personalized = {n: 1 for n in G.nodes}
    preferences = {
        '5000597': 5,
        '5000655': -3,
        '5000676': -100,
        '5000680': 7,
        '5000681': 2,
        '5000682': 4,
        '5000685': 2,
        '5000687': 6,
        '5000689': 6,
        '5000690': 100000,
        '5000693': 3,
        '5000696': 65,
    }

    personalized.update(preferences)
    print(f"size of graph before aggregate: {len(G.nodes) + len(G.edges)}")
    G_aggregated = NE_Aggregate_user_time__item(G)
    print(f"size of graph after aggregate: {len(G_aggregated.nodes) + len(G_aggregated.edges)}")
    G_adjusted = adjust_weights(G_aggregated, W_tf, W_cf)
    G_norm = normalize(G_adjusted)
    query_tau = 'movie'
    query_k = 5

    pr: dict = net.pagerank(G_norm, personalization=personalized, alpha=0.85, weight='weight')

    sorted_by_value: List[Any] = sorted(pr.items(), key=lambda kv: -kv[1])

    recommendations = []
    for sorted_element in sorted_by_value:
        if movie_id_offset < int(sorted_element[0]):
            recommendations.append(sorted_element)
            if len(recommendations) == query_k:
                break
    print(recommendations)


if __name__ == "__main__":
    main()
