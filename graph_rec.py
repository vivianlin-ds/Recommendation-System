# Method Description: The strategy here is to use a graph to build a recommendation system for the yelp data.

# Error Distribution:
# >=0 and <1: 102200
# >=1 and <2: 32910
# >=2 and <3: 6138
# >=3 and <4: 794
# >=4: 2

# RMSE:
# 0.9785272832228583

# Execution Time:
# 261.8420760631561

# from pyspark import SparkContext
import sys
import time
import json
from math import nan
import numpy as np
import networkx as nx
from node2vec import Node2Vec
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

start = time.time()

folder_path = sys.argv[1]
# test_file = sys.argv[2]
# output_file = sys.argv[3]

# sc = SparkContext('local[*]', 'graph_rec')
# sc.setLogLevel("ERROR")


def load_json_lines(file):
    data = []
    filepath = folder_path + file
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Read in the data first for businesses and users (the two groups of the bipartite graph)
print("Loading data")
bus = load_json_lines('/business.json')
users = load_json_lines('/user.json')
ratings = load_json_lines('/review_train.json')

print(len(bus), len(users), len(ratings))

# with open(folder_path + '/users.json', 'r') as f:
#     users = json.load(f)
#
# with open(folder_path + '/review_train.json', 'r') as f:
#     ratings = json.load(f)

# Build a bipartite graph
print("Building directed graph")
graph = nx.DiGraph()

# Add nodes for users and businesses
for user in users:
    graph.add_node(user['user_id'], bipartite=0, **user)
for b in bus:
    graph.add_node(b['business_id'], bipartite=1, **b)

for rating in ratings:
    user_id = rating['user_id']
    bus_id = rating['business_id']
    rate = rating['stars']

    # You can use the rating as the weight of the edge
    graph.add_edge(user_id, bus_id, weight=rate)

print("Graph completed!")

# Apply Node2Vec to generate embeddings
node2vec = Node2Vec(graph, dimensions=64, walk_length=10, num_walks=100, workers=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get embeddings for specific nodes
alice_embedding = model.wv['wf1GqnKQuvH-V3QN80UOOQ']
print("User's embedding:", alice_embedding)

business_embedding = model.wv['fThrN4tfupIGetkrz18JOg']
print("Business embedding:", business_embedding)

# Predict similarity (e.g., between Alice and Italian Restaurant)
similarity = model.wv.similarity('wf1GqnKQuvH-V3QN80UOOQ', 'fThrN4tfupIGetkrz18JOg')
print(f"Similarity: {similarity}")

# # Create a subgraph with edges having only 5 edges to make sure that the graph was successfully created
# import random
#
# # Randomly select 5 edges from the graph
# selected_edges = random.sample(list(graph.edges(data=True)), 5)
# # Create a subgraph with the selected edges
# subgraph_edges = [(u, v) for u, v, d in selected_edges]
# subgraph = graph.edge_subgraph(subgraph_edges).copy()
#
# import matplotlib.pyplot as plt
#
# # Draw the graph
# pos = nx.circular_layout(subgraph)
# nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
# edge_labels = nx.get_edge_attributes(subgraph, 'weight')
# nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red')
# plt.show()
#
# import matplotlib.pyplot as plt
#
# # Draw the graph
# pos = nx.circular_layout(graph)
# nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')
# edge_labels = nx.get_edge_attributes(graph, 'weight')
# nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
# plt.show()
#
# print("Starting recs")
# # Choose a user to make recommendations for
# # target_user = 'wf1GqnKQuvH-V3QN80UOOQ'
# target_business = 'd_Go5TsiFMIRtCE6XS8Sjw'
#
# # Set personalization: focus the random walk on the target user
# personalization = {node: 0 for node in graph.nodes}
# personalization[target_user] = 1
#
# # Compute the Personalized PageRank
# pr = nx.pagerank(graph, personalization=personalization, weight='weight')
#
# # Extract the score for Business3
# predicted_influence_score = pr['Business3']
#
# rated_businesses = list(graph.neighbors(target_user))
# print(len(rated_businesses))

# # Generate node embeddings (for simplicity, use node degrees as features)
# X = []
# y = []
# for u, v, data in graph.edges(data=True):
#     user_embedding = graph.degree[u]
#     business_embedding = graph.degree[v]
#
#     X.append([user_embedding, business_embedding])
#     y.append(data['weight'])
#
# X = np.array(X)
# y = np.array(y)
#
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)
#
# # Example prediction for a new user-business pair
# # Let's predict what rating Bob would give to the Italian Restaurant
# bob_embedding = graph.degree['wf1GqnKQuvH-V3QN80UOOQ']
# italian_restaurant_embedding = graph.degree['fThrN4tfupIGetkrz18JOg']
#
# new_data = np.array([[bob_embedding, italian_restaurant_embedding]])
# predicted_rating = model.predict(new_data)
# print("Predicted rating:", predicted_rating[0])
#
