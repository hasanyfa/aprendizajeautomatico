def nearest_neighbor(order_nodes, D, start='start', end='end'):
    unvisited = set(order_nodes)
    path = []
    curr = start
    while unvisited:
        nxt = min(unvisited, key=lambda x: D[curr][x])
        path.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    cost = D[start][path[0]] + sum(D[a][b] for a,b in zip(path, path[1:])) + D[path[-1]][end]
    return path, cost

def two_opt(path, D, start='start', end='end'):
    def route_cost(p):
        return D[start][p[0]] + sum(D[a][b] for a,b in zip(p, p[1:])) + D[p[-1]][end]
    improved = True
    best = path[:]
    best_cost = route_cost(best)
    while improved:
        improved = False
        for i in range(1, len(best)-1):
            for k in range(i+1, len(best)):
                new = best[:i] + best[i:k][::-1] + best[k:]
                c = route_cost(new)
                if c < best_cost:
                    best, best_cost = new, c
                    improved = True
                    break
            if improved: break
    return best, best_cost
