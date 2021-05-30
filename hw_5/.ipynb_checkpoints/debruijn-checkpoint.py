# Library for buildin De Bruijn graph by fasta/fastq file

import os
import networkx as nx
import pyfastx as fx
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import pydot
from copy import deepcopy
from itertools import product


def get_antisense(seq):
    """
    Get reversed complementary sequence
    """
    mapping = dict(zip(list("ACTG"), list("TGAC")))
    result = ""
    for nt in seq:
        result = mapping[nt] + result
    return result


def build_debruijn_graph(
    path_to_reads, kmer_len
):
    """
    Build De Bruijn graph. Adjacency matrix is returned
    Kmer abundance in reads is also provided
    """
    assert kmer_len % 2 == 1 
    reads = fx.Fastx(path_to_reads)
    kmer1_len = kmer_len + 1

    # DeBruijn graph we are growing
    dBGraph = nx.DiGraph()
    adjlist = {}

    # kmers coverage statistics
    kmer_coverage = {}
    def update_kmer_coverage(kmer):
        try:
            kmer_coverage[kmer] += 1
        except KeyError:
            kmer_coverage[kmer] = 1

    for read in tqdm(reads):
        seq = read[1]
        # check if read contains the valid symbols only
        assert set(read[1]) - set("ACTGN") == set()
        # scan the read by sliding window
        for i in range(len(seq)-kmer1_len+1):
            # kmers from reads
            edge = seq[i:i+kmer1_len]
            source = edge[:-1]
            target = edge[1:]
            update_kmer_coverage(edge)
            try:
                adjlist[source].add(target)
            except KeyError:
                adjlist[source] = {target}
            # reversed complementary kmers (the same sense)
            rc_edge = get_antisense(edge)
            rc_source = rc_edge[:-1]
            rc_target = rc_edge[1:]
            update_kmer_coverage(rc_edge)
            try:
                adjlist[rc_source].add(rc_target)
            except KeyError:
                adjlist[rc_source] = {rc_target}
    # adjacency lists return
    adjlist = {x: list(y) for x, y in adjlist.items()}
    return adjlist, kmer_coverage


def get_edges(adjlist):
    """
    Build edge table by adjacency matrix
    """
    edges = []
    for source, targets in adjlist.items():
        for s, t in product([source], targets):
            edges.append([s, t, s+t[-1]])
    return pd.DataFrame(edges, columns=["source", "target", "edge"])




def make_graph_compression_(edges_):
    """
    Perform De Bruijn graph compression
    """
    edges = deepcopy(edges_)
    kmer_len = len(edges.iloc[0, 0])
    source_nodes = edges.groupby("source")["edge"].count()
    source_nodes = source_nodes[source_nodes == 1].keys().to_list()
    target_nodes = edges.groupby("target")["edge"].count()
    target_nodes = target_nodes[target_nodes == 1].keys().to_list()
    uninformative_nodes = set(source_nodes) & set(target_nodes)
    # search through the graph until no uninformative node left
    for node in tqdm(uninformative_nodes):
        # select edges to merge
        merge = edges[
            (edges.target == node) | (edges.source == node)
        ]
        in_kmer = merge[merge.target == node].source.values[0]
        out_kmer = merge[merge.source == node].target.values[0]
        # drop them at first
        edges = edges[
            ~((edges.target == node) | (edges.source == node))
        ]
#         edges.drop(index=merge.index, inplace=True)
#         edges.reset_index(drop=True, inplace=True)
        # and introduce new merged edge
        new_edge = merge[merge.target == node].edge.values[0][:-kmer_len]+\
                   merge[merge.source == node].edge.values[0]
        new_edge = np.array([in_kmer, out_kmer, new_edge]).reshape(1, -1)
        edges = pd.concat([edges, pd.DataFrame(new_edge, columns=edges.columns)], axis=0)
    edges.reset_index(drop=True, inplace=True)
    return edges



def make_graph_compression(edges_):
    """
    Perform De Bruijn graph compression
    """
    edges = deepcopy(edges_)
    kmer_len = len(edges.iloc[0, 0])
    source_nodes = edges.groupby("source")["edge"].count()
    source_nodes = source_nodes[source_nodes == 1].keys().to_list()
    target_nodes = edges.groupby("target")["edge"].count()
    target_nodes = target_nodes[target_nodes == 1].keys().to_list()
    uninformative_nodes = set(source_nodes) & set(target_nodes)
    # search through the graph until no uninformative node left
    edges = edges.values
    for node in tqdm(uninformative_nodes):
        # select edges to merge
        drop_idxs = []
        drop_idxs.append(np.where(edges[:, 1] == node)[0][0])
        drop_idxs.append(np.where(edges[:, 0] == node)[0][0])
        merge = edges[drop_idxs, :]
        in_kmer = merge[0, 0]
        out_kmer = merge[1, 1]
        # drop them at first
        edges = np.delete(edges, drop_idxs, axis=0)
        # and introduce new merged edge
        new_edge = merge[0, 2][:-kmer_len]+\
                   merge[1, 2]
        new_edge = np.array([in_kmer, out_kmer, new_edge]).reshape(1, -1)
        edges = np.concatenate([edges, new_edge], axis=0)
    edges = pd.DataFrame(edges, columns=["source", "target", "edge"])
    return edges



def get_mean_coverage(edge, kmer_len, kmer_coverage):
    """
    Mean edge coverage calculation
    """
    cov_list = []
    for i in range(len(edge)-kmer_len):
        kmer = edge[i:i+kmer_len+1]
        cov_list.append(kmer_coverage[kmer])
#         try:
#             cov_list.append(kmer_coverage[kmer])
#         except KeyError:
#             cov_list.append(1)
    return np.mean(cov_list)


def add_edges_statistics(edges_, kmer_len, kmer_coverage):
    """
    Enrich edge table with addition statistics
    """
    edges = deepcopy(edges_)
    edges["length"] = edges["edge"].apply(len)
    edges["mean_cov"] = edges["edge"].apply(lambda x: get_mean_coverage(x, kmer_len, kmer_coverage))
    return edges


def remove_tips(edges_, cov_cutoff, len_cutoff):
    """
    Remove tips from builded De Bruijn graph
    """
    edges = deepcopy(edges_)
    source_dict = dict(edges.groupby("source")["edge"].count())
    target_dict = dict(edges.groupby("target")["edge"].count())
    nodes = set(source_dict.keys()) | set(target_dict.keys())
    for node in nodes:
        try:
            in_degree = target_dict[node]
        except KeyError:
            in_degree = 0
        try:
            out_degree = source_dict[node]
        except KeyError:
            out_degree = 0
        if in_degree + out_degree == 1: # dead end condition
            edge = edges[(edges.source == node) | (edges.target == node)]
            if edge.length.values[0] < len_cutoff or edge.mean_cov.values[0] < cov_cutoff:
                # remove edge if it doesn't meet required cut-offs
                edges = edges[~((edges.source == node) | (edges.target == node))]
                edges.reset_index(drop=True, inplace=True)
                source_dict = dict(edges.groupby("source")["edge"].count())
                target_dict = dict(edges.groupby("target")["edge"].count())
    return edges


def remove_any(edges_, cov_cutoff, len_cutoff):
    """
    Remove tips from builded De Bruijn graph
    """
    edges = deepcopy(edges_)
    while True:
        for i, edge in edges.iterrows():   
            if edge.length < len_cutoff or edge.mean_cov < cov_cutoff:
                # remove edge if it doesn't meet required cut-offs
                edges.drop(index=i, inplace=True)
                edges.reset_index(drop=True, inplace=True)
                break
        else:
            break
    return edges


def save_dot(edges, res_path, dot_file, nodes_file):
    """
    Graph saving if DOT format. Auxiliary JSON with nodes indicies is also dumped
    """
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    path_to_dot = os.path.join(res_path, dot_file)
    path_to_nodes = os.path.join(res_path, nodes_file)
    with open(path_to_dot, "w") as dot:
        dot.write("digraph {\n")
        nodes = set(edges.source.values) | set(edges.target.values)
        node_id = {node: i for i, node in enumerate(nodes)}
        json.dump(dict(enumerate(nodes)), open(path_to_nodes, "w"))
        for node in nodes:
            dot.write(f"{node_id[node]};\n")
        for i, (source, target) in enumerate(zip(edges.source, edges.target)):
            label = f"Len:{edges.length[i]};Cov:{edges.mean_cov[i]:.1f}"
            dot.write(f"""{node_id[source]} -> {node_id[target]} [label="{label}"];\n""") 
        dot.write("}")


def dot2png(res_path, dot_file, png_file):
    """
    Create graph visualisation in PNG by given DOT file
    """
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    path_to_png = os.path.join(res_path, png_file)
    path_to_dot = os.path.join(res_path, dot_file)
    (graph,) = pydot.graph_from_dot_file(path_to_dot)
    graph.write_png(path_to_png)
    

def edges2fasta(edges, res_path, fasta_file):
    """
    Dump assembled edges to FASTA file
    """
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    path_to_fasta = os.path.join(res_path, fasta_file)
    with open(path_to_fasta, "w") as fasta:
        for i, row in edges.iterrows():
            title = f"> {row.source} -> {row.target} | Len:{row.length};Cov:{row.mean_cov:.1f}"
            seq = row.edge
            fasta.write(f"{title}\n{seq}\n")