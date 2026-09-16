"""Fixed graph controls with explicit row budgets."""

import networkx as nx

import numpy as np


def sample_random_sparse_matched_row_degree(mask, d_star, graph_seed):
    mask = np.asarray(mask) > 0
    d_star = np.asarray(d_star, dtype=np.int32)
    rng = np.random.default_rng(int(graph_seed))
    N = mask.shape[0]
    z_rand = np.zeros((N, N), dtype=np.uint8)

    for i in range(N):
        cand = np.where(mask[i] > 0)[0]
        d_i = int(d_star[i])
        if d_i > len(cand):
            raise ValueError(
                f"Row {i}: cannot sample {d_i} edges from {len(cand)} candidates."
            )
        if d_i > 0:
            choice = rng.choice(cand, size=d_i, replace=False)
            z_rand[i, np.asarray(choice, dtype=np.int32)] = 1

    actual = np.asarray(z_rand.sum(axis=1), dtype=np.int32)
    if not np.array_equal(actual, d_star):
        raise ValueError(
            "Exact row-degree match failed for sampled random sparse graph."
        )

    return z_rand


def build_heuristic_shortest_path_budgeted_graph(problem, cfg_run):
    M = np.asarray(problem["M"])
    L = np.asarray(problem["L"])
    reachable_pairs = np.asarray(problem["reachable_pairs"], dtype=np.int32)
    N = M.shape[0]
    k_router = int(cfg_run["k_router"])
    open_budget_topm = int(cfg_run["open_budget_topm"])

    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        cand = np.where(M[i] > 0)[0]
        for j in np.asarray(sorted(cand), dtype=np.int32):
            G.add_edge(int(i), int(j), weight=float(L[i, j]))

    edge_count = np.zeros((N, N), dtype=np.int32)
    for s, t in reachable_pairs:
        try:
            path = nx.shortest_path(
                G, source=int(s), target=int(t), weight="weight", method="dijkstra"
            )
        except nx.NetworkXNoPath:
            continue
        for u, v in zip(path[:-1], path[1:]):
            edge_count[int(u), int(v)] += 1

    row_score = edge_count.sum(axis=1)
    valid_rows = [i for i in range(N) if np.any(M[i] > 0)]
    open_rows_sorted = sorted(
        valid_rows,
        key=lambda i: (
            -int(row_score[i]),
            float(np.min(L[i, np.where(M[i] > 0)[0]])),
            int(i),
        ),
    )
    k_open = min(open_budget_topm, len(valid_rows))
    open_rows = set(open_rows_sorted[:k_open])

    z_graph = np.zeros((N, N), dtype=np.uint8)

    for i in open_rows:
        cand = np.where(M[i] > 0)[0]
        ordered = sorted(
            [int(j) for j in cand],
            key=lambda j: (-int(edge_count[i, j]), float(L[i, j]), int(j)),
        )
        chosen = []
        for j in ordered:
            if edge_count[i, j] > 0 and len(chosen) < k_router:
                chosen.append(int(j))
        if len(chosen) < k_router:
            for j in sorted(
                [int(j) for j in cand], key=lambda j: (float(L[i, j]), int(j))
            ):
                if j not in chosen:
                    chosen.append(int(j))
                if len(chosen) == k_router:
                    break
        for j in chosen[:k_router]:
            z_graph[i, j] = 1

    out_degree = z_graph.sum(axis=1)
    if int((out_degree > 0).sum()) > open_budget_topm:
        raise ValueError("Heuristic graph violates open_budget_topm.")
    if np.any(out_degree > k_router):
        raise ValueError("Heuristic graph violates k_router.")

    return z_graph
