from __future__ import annotations
from dataclasses import dataclass
import math
import pandas as pd
import networkx as nx


@dataclass
class PortGraph:
    graph: nx.DiGraph
    node_master: pd.DataFrame
    link_master: pd.DataFrame


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_port_graph(anchor_master: pd.DataFrame, max_quay_nodes: int = 18) -> PortGraph:
    """Create a runnable terminal road graph from TOS Bitt/Berth anchors.

    Real TOS JSON has Bitt/Berth anchors only. QC/Yard/Gate/Intersection nodes
    must therefore be supplied by DT map or manually defined Node Master. This
    function creates a deterministic Node Master from the anchor range for pre-DT
    algorithm validation.
    """
    anchors = anchor_master.sort_values("x").reset_index(drop=True)
    step = max(1, len(anchors) // max_quay_nodes)
    quay_anchors = anchors.iloc[::step].copy().reset_index(drop=True)
    if len(quay_anchors) < 3:
        quay_anchors = anchors.copy()

    x_min, x_max = float(anchors["x"].min()), float(anchors["x"].max())
    y_base = float(anchors["y"].median())
    x_span = x_max - x_min

    nodes: list[dict] = []

    def add_node(node_id: str, node_type: str, x: float, y: float, berth: str = "", name: str = ""):
        nodes.append({
            "node_id": node_id,
            "node_type": node_type,
            "x": float(x),
            "y": float(y),
            "berth": berth,
            "name": name or node_id,
        })

    # Quay / QC / road intersections aligned to actual Bitt anchors.
    for i, row in quay_anchors.iterrows():
        x = float(row["x"])
        berth = str(row["berth"])
        add_node(f"Q{i:02d}", "QUAY_ANCHOR", x, y_base, berth, f"{berth}-{row['bitt']}")
        add_node(f"QC{i:02d}", "QC", x, y_base + 22, berth, f"QC_{i:02d}")
        add_node(f"R{i:02d}", "ROAD_INTERSECTION", x, y_base + 85, berth, f"R_{i:02d}")

    # Yard blocks as rows behind the quay.
    yard_rows = [170, 255, 340]
    yard_cols = 6
    for r, y_off in enumerate(yard_rows):
        for c in range(yard_cols):
            x = x_min + (c + 0.5) * x_span / yard_cols
            add_node(f"Y{r}{c}", "YARD_BLOCK", x, y_base + y_off, "", f"YARD_{chr(65+r)}{c+1}")

    # Gate nodes.
    add_node("G_IN", "GATE", x_min - 90, y_base + 255, "", "GATE_IN")
    add_node("G_OUT", "GATE", x_max + 90, y_base + 255, "", "GATE_OUT")

    node_df = pd.DataFrame(nodes)
    pos = {r.node_id: (float(r.x), float(r.y)) for r in node_df.itertuples()}

    links: list[dict] = []

    def add_edge(a: str, b: str, road_type: str = "ROAD", speed: float = 7.0, capacity: int = 4, bidirectional: bool = True):
        for u, v in [(a, b)] + ([(b, a)] if bidirectional else []):
            length = _dist(pos[u], pos[v])
            link_id = f"L{len(links):04d}"
            # weight is base travel time in seconds.
            links.append({
                "link_id": link_id,
                "from_node": u,
                "to_node": v,
                "road_type": road_type,
                "length_m": length,
                "speed_limit_mps": speed,
                "capacity": capacity,
                "base_time_s": length / max(speed, 0.1),
            })

    n_quay = len(quay_anchors)
    for i in range(n_quay - 1):
        add_edge(f"Q{i:02d}", f"Q{i+1:02d}", "QUAY_AXIS", 5.0, 3)
        add_edge(f"R{i:02d}", f"R{i+1:02d}", "MAIN_ROAD", 8.0, 6)
    for i in range(n_quay):
        add_edge(f"Q{i:02d}", f"QC{i:02d}", "QC_ACCESS", 4.5, 2)
        add_edge(f"QC{i:02d}", f"R{i:02d}", "APRON_ACCESS", 5.5, 3)

    # Yard row internal links and vertical connectors to nearest road intersections.
    for r in range(3):
        for c in range(yard_cols - 1):
            add_edge(f"Y{r}{c}", f"Y{r}{c+1}", "YARD_LANE", 5.5, 3)
        for c in range(yard_cols):
            if r < 2:
                add_edge(f"Y{r}{c}", f"Y{r+1}{c}", "YARD_CONNECTOR", 4.5, 2)
            y_node = f"Y{r}{c}"
            nearest_i = min(range(n_quay), key=lambda i: abs(pos[f"R{i:02d}"][0] - pos[y_node][0]))
            add_edge(y_node, f"R{nearest_i:02d}", "YARD_ACCESS", 5.0, 3)

    add_edge("G_IN", "Y10", "GATE_ACCESS", 4.5, 2)
    add_edge("G_IN", "Y20", "GATE_ACCESS", 4.5, 2)
    add_edge("G_OUT", f"Y1{yard_cols-1}", "GATE_ACCESS", 4.5, 2)
    add_edge("G_OUT", f"Y2{yard_cols-1}", "GATE_ACCESS", 4.5, 2)

    link_df = pd.DataFrame(links)
    G = nx.DiGraph()
    for row in node_df.itertuples(index=False):
        G.add_node(row.node_id, node_type=row.node_type, x=row.x, y=row.y, name=row.name)
    for row in link_df.itertuples(index=False):
        G.add_edge(row.from_node, row.to_node, link_id=row.link_id, weight=row.base_time_s,
                   length_m=row.length_m, capacity=row.capacity, speed_limit_mps=row.speed_limit_mps,
                   road_type=row.road_type)
    return PortGraph(G, node_df, link_df)


def adjacency_from_links(link_master: pd.DataFrame) -> pd.DataFrame:
    """Build link-to-link adjacency matrix for ST-GNN.

    Two directed links are adjacent when one link's to_node equals the other's from_node.
    """
    links = link_master.reset_index(drop=True)
    ids = links["link_id"].tolist()
    idx = {lid: i for i, lid in enumerate(ids)}
    A = pd.DataFrame(0.0, index=ids, columns=ids)
    for a in links.itertuples(index=False):
        for b in links.itertuples(index=False):
            if a.to_node == b.from_node:
                A.loc[a.link_id, b.link_id] = 1.0
            if a.link_id == b.link_id:
                A.loc[a.link_id, b.link_id] = 1.0
    return A


def k_shortest_paths(G: nx.DiGraph, origin: str, dest: str, k: int = 5) -> list[list[str]]:
    """Fast practical K candidate paths for the validation project.

    NetworkX `shortest_simple_paths` can be slow on bidirectional terminal
    graphs with many cycles. For this pre-Meta K-Port validator we generate
    diverse candidates by repeatedly running shortest path with temporary
    penalties on links already used by earlier candidates. The result is not a
    strict Yen K-shortest implementation, but it is fast, deterministic, and
    appropriate for candidate-route/baseline generation.
    """
    paths: list[list[str]] = []
    if origin == dest:
        return []
    try:
        # preserve original weights
        original = {(u, v): float(d.get("weight", 1.0)) for u, v, d in G.edges(data=True)}
        for cand in range(k):
            path = nx.shortest_path(G, origin, dest, weight="weight")
            if path not in paths:
                paths.append(path)
            # Penalize all edges in the selected path to expose alternatives.
            factor = 1.35 + 0.25 * cand
            for u, v in zip(path[:-1], path[1:]):
                G[u][v]["weight"] = float(G[u][v].get("weight", 1.0)) * factor + 8.0
        # restore
        for (u, v), w in original.items():
            G[u][v]["weight"] = w
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return paths
    finally:
        # Safety restore if an exception occurs after partial penalty.
        if 'original' in locals():
            for (u, v), w in original.items():
                if G.has_edge(u, v):
                    G[u][v]["weight"] = w


def path_to_link_ids(G: nx.DiGraph, path: list[str]) -> list[str]:
    link_ids = []
    for u, v in zip(path[:-1], path[1:]):
        link_ids.append(G[u][v]["link_id"])
    return link_ids
