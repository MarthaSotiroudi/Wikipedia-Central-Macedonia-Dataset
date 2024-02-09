import pandas as pd
import networkx as nx
import os
import numpy as np
from graph import Graph
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import SparsePCA, PCA, KernelPCA, TruncatedSVD
import torch
import json
from collections import Counter

import torch


raw_input_dir = '../workspace/wikispeedia/raw_input_data/'
output_dir = '/media/data/marthass/wikispeedia-paths-dual-hypergraph-features-main/gretel/'
input_dir = '/media/data/marthass/wikispeedia-paths-dual-hypergraph-features-main/gretel/'

"""
Load dataframes

"""

import re


import pandas as pd
import networkx as nx


import pandas as pd

import torch
print(torch.cuda.is_available())



with open("articles.tsv", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Remove or replace stray quotes
cleaned_lines = [line.replace('"', '') for line in lines]

with open("articles_cleaned.tsv", "w", encoding="utf-8") as file:
    file.writelines(cleaned_lines)

# Now read the cleaned file
articles_df = pd.read_csv('articles_cleaned.tsv', delimiter='\t', header=None, names=['article'])

# Ensure the DataFrame is loaded correctly by displaying the first few rows
print(articles_df.head())
articles_df.drop_duplicates(subset='article', inplace=True)



import pandas as pd

paths = []
links = []
with open('3000.tsv', 'r') as f:
    current_path = []
    for line in f:
        articles = line.strip().split('\t')

        # Check if the line starts a new path
        if articles[0] == "Central_Macedonia":
            # If we already have a path, append it to paths and start a new one
            if current_path:
                paths.append(current_path)
                current_path = []

        current_path.extend(articles)


    # Handle the last path
    if current_path:
        paths.append(current_path)






# Convert paths to DataFrame
paths_df = pd.DataFrame(paths)

import pandas as pd



print(paths_df.head())

# Rename the first column to 'Source' and the rest to 'Destination_1', 'Destination_2', etc.
paths_df.columns = ['Source'] + [f'Destination_{i}' for i in range(1, paths_df.shape[1])]

# Iterating over each path (row) in paths_df
all_paths = []

# Iterating over each path (row) in paths_df
for _, row in paths_df.iterrows():
    path = [row['Source']]
    for col in paths_df.columns:
        if col.startswith('Destination_') and pd.notnull(row[col]):
            path.append(row[col])
    all_paths.append(path)

# Create an empty list for storing edges
edges = []

# Iterate through each row in paths_df to create edges
for _, row in paths_df.iterrows():
    path = row.dropna().values  # Get the non-null articles in the path
    # Create edges between consecutive articles in the path
    for i in range(len(path) - 1):
        fr = path[i]
        to = path[i + 1]
        edges.append((fr, to))  # Add the edge to the list

# Convert the edges list to a DataFrame
links_df = pd.DataFrame(edges, columns=['linkSource', 'linkTarget'])

# Remove potential duplicates if necessary
links_df = links_df.drop_duplicates().reset_index(drop=True)

# Save links_df as "links.tsv"
links_df.to_csv('links2.tsv', sep='\t', index=False, header=False)


# Print the first few rows of links_df to verify
print(links_df.head())

# Now, all_paths contains a list of paths, with each path being a list of articles
print(all_paths)

cleaned_paths = []

for path in all_paths:
    cleaned_path = [article.replace('"', '').strip() for article in path]
    cleaned_paths.append(cleaned_path)

print(cleaned_paths)

# Construct paths
paths_df['path'] = paths_df.apply(lambda row: [article for article in row if pd.notnull(article)], axis=1)



# 1. Cleaning links_df
links_df['linkSource'] = links_df['linkSource'].str.replace('"', '')
links_df['linkTarget'] = links_df['linkTarget'].str.replace('"', '')

# 2. Cleaning paths_df
# Assuming 'path' is the column name containing lists of article paths
paths_df['path'] = paths_df['path'].apply(lambda path_list: [article.replace('"', '') for article in path_list])

# If paths are stored in separate columns (like 'Destination_1', 'Destination_2', ...),
# you'll need to iterate over these columns:
for col in paths_df.columns:
    paths_df[col] = paths_df[col].str.replace('"', '')

# Display the first few rows to ensure it's loaded correctly
print(paths_df.head())
print(paths_df.columns)

# Print the first few rows of the DataFrame
print(links_df.head())

# Print the last few rows of the DataFrame
print(links_df.tail())


# If the DataFrame is too large and you only want to see a sample, you can use:
print(links_df.sample(5))




article_to_id = {a: i for i, a in enumerate(articles_df.article.values)}
id_to_article = {i: a for i, a in enumerate(articles_df.article.values)}


articles_df.drop_duplicates(subset='article', inplace=True)

if links_df.empty:
    raise ValueError("The links_df DataFrame is empty. Cannot create edge_ids.")

missing_articles = set(links_df['linkSource']).union(set(links_df['linkTarget'])) - set(article_to_id.keys())
if missing_articles:
    print(f"Missing article mappings: {missing_articles}")
    # You might need to add the missing articles to the article_to_id mapping

# Assuming links_df is a DataFrame with columns 'linkSource' and 'linkTarget'
if 'linkSource' not in links_df.columns or 'linkTarget' not in links_df.columns:
    raise ValueError("links_df does not contain the required columns.")

# Create edge_ids mapping
edge_ids = {(article_to_id[src], article_to_id[trg]): idx for idx, (src, trg) in enumerate(zip(links_df['linkSource'], links_df['linkTarget'])) if src in article_to_id and trg in article_to_id}

if not edge_ids:
    raise ValueError("No edges could be created from links_df. Check the data and mappings.")



paths_df = paths_df.reset_index()

DEFAULT_ID = max(article_to_id.values()) + 1


def get_article_id(article):
    """Retrieve the article ID from the dictionary. If not found, return DEFAULT_ID."""
    return article_to_id.get(article, DEFAULT_ID)


def clean_article_name(article_name):
    """Remove extraneous characters like quotes from the article name."""
    return article_name.replace('"', '').strip()

# Apply cleaning to the paths
cleaned_paths = [[clean_article_name(article) for article in path] for path in paths]

# Testing if all links in cleaned paths are in edge_ids after cleaning
missing_links_after_cleaning = set()
for path in cleaned_paths:
    for fr, to in zip(path, path[1:]):
        try:
            fr_id = get_article_id(fr)
            to_id = get_article_id(to)
            edge_index = edge_ids[(fr_id, to_id)]
        except KeyError:
            missing_links_after_cleaning.add((fr, to))

print("Missing links in cleaned paths after targeted cleaning:", missing_links_after_cleaning)

# Convert cleaned_paths to a DataFrame
paths_df = pd.DataFrame(cleaned_paths)

# Depending on your requirements, you might want to rename the columns
paths_df.columns = ['Source'] + [f'Destination_{i}' for i in range(1, paths_df.shape[1])]





n_nodes = len(article_to_id)
n_edges = len(edge_ids)

n_hypernodes = n_edges
n_hyperedges = n_nodes


original_node_features = {}
with open(os.path.join(input_dir, "nodes.txt")) as f:
    first_line = f.readline()
    for line in f:
        feat = []
        num_nodes = int(f.readline().strip().split("\t")[0])

    # Now read the rest of the file for node features
    for i, line in enumerate(f.readlines()):  # 'i' starts at 0, so line numbers will be offset by one
        parts = line.strip().split("\t")
        if len(parts) < 3:
            print(f"Warning: Line {i + 2} is malformed: {line}")
        node_id, out_d, in_d = int(parts[0]), float(parts[1]), float(parts[2])
        node_label = articles_df.iloc[node_id][0]
        original_node_features[node_id] = {}
        original_node_features[node_id]['label'] = node_label
        original_node_features[node_id]['out_degree'] = out_d
        original_node_features[node_id]['in_degree'] = in_d

original_edge_features = {}
with open(os.path.join(input_dir, "edges.txt")) as f:
    first_line = f.readline()
    for line in f:
        feat = []
        line = line.strip().split("\t")
        edge_id, out_id, in_id, tfidf, nof = int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4])
        original_edge_features[edge_id] = {}
        original_edge_features[edge_id]['out_id'] = out_id
        original_edge_features[edge_id]['in_id'] = in_id
        original_edge_features[edge_id]['tfidf'] = tfidf
        original_edge_features[edge_id]['nof'] = nof


if links_df.empty:
    raise ValueError("The links DataFrame is empty. Cannot create edge_ids.")

missing_articles = set(links_df['linkSource']).union(set(links_df['linkTarget'])) - set(article_to_id.keys())
if missing_articles:
    print(f"Missing articles in article_to_id mapping: {missing_articles}")
    # Handle the missing articles (e.g., add them to article_to_id or correct the data)





# Define the compute_incidence_matrix function
def compute_incidence_matrix(edges_dict, num_nodes):
    num_edges = len(edges_dict)
    incidence_matrix = np.zeros((num_nodes, num_edges))
    for (source, target), edge_id in edges_dict.items():
        incidence_matrix[source, edge_id] = 1
        incidence_matrix[target, edge_id] = 1
    return incidence_matrix


n_nodes = len(article_to_id)  # Number of nodes should match the number of unique articles
n_edges = len(edge_ids)  # Number of edges should match the number of unique edges

# The incidence matrix should have dimensions of (n_nodes, n_edges)
graph_incidence = compute_incidence_matrix(edge_ids, n_nodes)

# Check if the incidence matrix is correctly populated
if graph_incidence.shape[1] == 0:
    raise ValueError("graph_incidence matrix has 0 features. Cannot fit the model.")

hypergraph_incidence = np.transpose(graph_incidence)

# Verify that the incidence matrix has been computed correctly
if graph_incidence.shape == (n_nodes, 0):
    raise ValueError("graph_incidence matrix has 0 features. Cannot fit the model.")




x = hypergraph_incidence.sum(axis=0).astype(int)   # check how many hypernodes are related with each hyperedge # x = [11, 19, 20, 8, 10, 47, 91....]
max(x)  # 1845
# Check if graph_incidence has any features
if graph_incidence.shape[1] == 0:
    raise ValueError("graph_incidence has 0 features. Cannot fit the model.")

print("Shape of graph_incidence:", graph_incidence.shape)
print("Content of graph_incidence:", graph_incidence)



transformer = KernelPCA(n_components=50, random_state=0)
transformer.fit(graph_incidence)
graph_incidence_transformed = transformer.transform(graph_incidence)


# den exei ginei normalization alla einai san kanonikopoiimena
with open(os.path.join(input_dir, 'edges_original_similarity_hyperedge_KernelPCA_features2.txt'), 'w') as f:
    f.write("{}\t{}\n".format(n_edges, 3))
    for i in range(0, n_edges):
        print(i)
        feat = []
        fr = original_edge_features[i]['out_id']
        to = original_edge_features[i]['in_id']
        hyperedge_fr = graph_incidence[fr, :].reshape(1, -1)
        hyperedge_to = graph_incidence[to, :].reshape(1, -1)
        sim = cosine_similarity(hyperedge_fr, hyperedge_to).reshape(-1)
        feat.append(fr)
        feat.append(to)
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(sim[0])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)

######### CHECK  NORMALIZED SIMILARITY ###############
similarity = []
for i in range(0, n_edges):
    print(i)
    fr = original_edge_features[i]['out_id']
    to = original_edge_features[i]['in_id']
    hyperedge_fr = graph_incidence[fr, :].reshape(1, -1)
    hyperedge_to = graph_incidence[to, :].reshape(1, -1)
    sim = cosine_similarity(hyperedge_fr, hyperedge_to).reshape(-1)
    similarity.append(sim[0])

# max(similarity) = 1.0000000000000004
# min(similarity) = 0.0006895240031534343
# no extra normalization is needed

############################################################################################################
############################################################################################################


with open(os.path.join(input_dir, 'edges_original_similarity_hyperedge_KernelPCA_features2.txt'), 'r') as f:
    first_line = f.readline()
    for line in f:
        feat = []
        line = line.strip().split("\t")
        edge_id, out_id, in_id, tfidf, nof, similariry = int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4]),float(line[5])
        original_edge_features[edge_id]['similariry'] = similariry


############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
def compute_directed_incidence_matrix(edges, n_nodes):
    n_edges = len(edges)
    incidence = np.zeros((n_nodes, n_edges))
    f = 0
    t = 0
    for edge_id, edge in enumerate(edges):
        # print(edge_id, edge)
        if incidence[edge[0], edge_id] == 0:
            incidence[edge[0], edge_id] = -1
        else:
            print('mpika')
        if incidence[edge[1], edge_id] == 0:
            incidence[edge[1], edge_id] = 1
        else:
            if incidence[edge[1], edge_id] == -1:
                f += 1
            else:
                t += 1
    print (f, t)
    return incidence
############################################################################################################
############################################################################################################
############################################################################################################
############### COMPUTE IN - OUT DEGREE FOR EACH HYPERNODE ###########

graph_incidence = compute_directed_incidence_matrix(edge_ids, n_nodes)
hypergraph_incidence = np.transpose(graph_incidence)

hypersenders = []
hyperreceivers = []
for i in range(0, n_hyperedges):
    hyperedge = hypergraph_incidence[:, i]
    out_list = np.where(hyperedge == -1)[0]
    in_list = np.where(hyperedge == 1)[0]
    if len(in_list) > 0 and len(out_list) > 0:
        for in_element in in_list:
            for out_element in out_list:
                hypersenders.append(in_element)
                hyperreceivers.append(out_element)

in_degree = np.zeros(n_edges)
out_degree = np.zeros(n_edges)
nof_hypersenders = Counter(hypersenders)
nof_hyperreceivers = Counter(hyperreceivers)

for i in range(0, n_edges):
    in_degree[i] = nof_hyperreceivers[i]
    out_degree[i] = nof_hypersenders[i]

in_degree = in_degree / max(in_degree)
out_degree = out_degree / max(out_degree)

with open(os.path.join(input_dir, 'edges_original_hyperedge_in_out_degree2.txt'), 'w') as f:
    f.write("{}\t{}\n".format(n_edges, 4))
    for i in range(0, n_edges):
        print(i)
        feat = []
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(in_degree[i])
        feat.append(out_degree[i])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)


############################################################################################################
with open(os.path.join(input_dir, 'edges_original_similarity_hyperedge_in_out_degree2.txt'), 'w') as f:
    f.write("{}\t{}\n".format(n_edges, 5))
    for i in range(0, n_edges):
        print(i)
        feat = []
        feat.append(original_edge_features[i]['out_id'])
        feat.append(original_edge_features[i]['in_id'])
        feat.append(original_edge_features[i]['tfidf'])
        feat.append(original_edge_features[i]['nof'])
        feat.append(similarity[i])
        feat.append(in_degree[i])
        feat.append(out_degree[i])
        line = str(i) + "\t" + "\t".join(map(str, [f for f in feat])) + "\n"
        f.write(line)
