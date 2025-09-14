import networkx as nx

G = nx.DiGraph()
# G.add_edge(u, v, weight=seconds)  # construir desde plano de tienda
# ...

def shortest_time(u, v):
    return nx.shortest_path_length(G, u, v, weight='weight')

P = ['start'] + item_nodes + ['end']
# Matriz D de tiempos entre todos los nodos de P:
D = {a: {b: shortest_time(a,b) for b in P if b != a} for a in P}
