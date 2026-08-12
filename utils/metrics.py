"""
metrics.py - Graph quality metrics for Tesseract++ navigable graphs

Definitions (stated precisely so results are reproducible):

  completeness            fraction of unordered MAIN-room pairs connected by
                          a path (1.0 = every room reaches every room)
  exit_reachability       fraction of main rooms with a path to at least one
                          exit door
  transition_reachability fraction of main rooms with a path to at least one
                          floor transition (stairs/elevator); None when the
                          graph has no transitions
  components              connected-component count and largest-component
                          share, over the FULL graph (exhaustive, not
                          sampled)

Main rooms are nodes of type 'room' whose id contains no '_subnode_'.
Exit doors are nodes whose (original) id starts with 'exit_door_'.
Works on single-floor graphs and merged multi-floor graphs (whose node ids
are floor-prefixed and carry original_id).
"""

import json

import networkx as nx


def load_graph_json(path):
    with open(path) as f:
        data = json.load(f)
    G = nx.Graph()
    for nd in data['nodes']:
        G.add_node(nd['id'], **{k: v for k, v in nd.items() if k != 'id'})
    for ed in data['edges']:
        G.add_edge(ed['source'], ed['target'],
                   weight=float(ed.get('weight') or 1.0))
    return G


def original_id(G, n):
    return G.nodes[n].get('original_id') or str(n)


def main_rooms(G):
    return [n for n, d in G.nodes(data=True)
            if d.get('type') == 'room' and '_subnode_' not in str(n)]


def exit_doors(G):
    return [n for n in G if original_id(G, n).startswith('exit_door_')]


def transitions(G):
    return [n for n, d in G.nodes(data=True)
            if d.get('type') == 'floor_transition']


def _component_index(G):
    comp_of = {}
    for ci, comp in enumerate(nx.connected_components(G)):
        for n in comp:
            comp_of[n] = ci
    return comp_of


def compute_metrics(G):
    """All metrics in one pass (component decomposition, no per-pair
    Dijkstra needed - connectivity is a component property)."""
    rooms = main_rooms(G)
    exits = exit_doors(G)
    trans = transitions(G)
    comp_of = _component_index(G)

    # completeness over main-room pairs
    from collections import Counter
    room_comp_sizes = Counter(comp_of[r] for r in rooms)
    n = len(rooms)
    connected_pairs = sum(k * (k - 1) // 2 for k in room_comp_sizes.values())
    total_pairs = n * (n - 1) // 2
    completeness = connected_pairs / total_pairs if total_pairs else 1.0

    exit_comps = {comp_of[e] for e in exits}
    exit_reach = (sum(1 for r in rooms if comp_of[r] in exit_comps) / n
                  if n and exits else (None if not exits else 1.0))

    trans_comps = {comp_of[t] for t in trans}
    trans_reach = (sum(1 for r in rooms if comp_of[r] in trans_comps) / n
                   if n and trans else None)

    comp_sizes = Counter(comp_of.values())
    largest = max(comp_sizes.values()) if comp_sizes else 0

    return {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_main_rooms': n,
        'n_exit_doors': len(exits),
        'n_transitions': len(trans),
        'completeness': completeness,
        'exit_reachability': exit_reach,
        'transition_reachability': trans_reach,
        'n_components': len(comp_sizes),
        'largest_component_share': (largest / G.number_of_nodes()
                                    if G.number_of_nodes() else 0.0),
        'room_components': len(room_comp_sizes),
    }


def format_metrics(m, label=""):
    def pct(v):
        return "n/a" if v is None else f"{v:.3f}"
    return (f"{label:>18}  nodes={m['n_nodes']:>5} edges={m['n_edges']:>5} "
            f"rooms={m['n_main_rooms']:>4} exits={m['n_exit_doors']:>2} "
            f"trans={m['n_transitions']:>2} | completeness={pct(m['completeness'])} "
            f"exit_reach={pct(m['exit_reachability'])} "
            f"trans_reach={pct(m['transition_reachability'])} "
            f"components={m['n_components']}")
