import pickle
import networkx as nx

with open("../data/graph_complete.pkl", "rb") as f:
    G = pickle.load(f)

largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc).copy()

with open("../data/graph_lcc.pkl", "wb") as f:
    pickle.dump(G_lcc, f)

print("Nœuds LCC:", G_lcc.number_of_nodes())
print("Arêtes LCC:", G_lcc.number_of_edges())
