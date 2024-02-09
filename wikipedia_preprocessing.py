import pandas as pd
import networkx as nx
import os
import numpy as np
from graph import Graph
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDNN Version: {torch.backends.cudnn.version()}")
print(f"Is GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")



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


import pandas as pd

# Read in the articles file and clean it
articles_df = pd.read_csv('unique_articles.tsv', delimiter='\t', header=None, names=['article'])
articles_df['article'] = articles_df['article'].str.replace('"', '')
articles_df.drop_duplicates(subset='article', inplace=True)

# Create the mappings

article_to_id = {a: i for i, a in enumerate(articles_df['article'])}
id_to_article = {i: a for i, a in enumerate(articles_df['article'])}

# Now read the 3000.tsv file
# Initialize an empty list to store the paths
paths = []

# Open the 3000.tsv file and read line by line
with open('3000_routes.csv', 'r') as file:
    # Initialize an empty list for the current path
    current_path = []
    
    # Iterate over each line in the file
    for line in file:
        # Strip whitespace and split the line by tab to get the articles
        articles = line.strip().split('\t')
        
        # Check if the first article in the line is 'Central_Macedonia', 
        # which signifies the start of a new path
        if articles[0] == "Central_Macedonia":
            # If current_path is not empty, we've reached a new path; 
            # so save the current_path to paths and start a new one
            if current_path:
                paths.append(current_path)
                current_path = []
        
        # Extend the current path with the articles in the current line
        current_path.extend(articles)
    
    # After the last line, add the last path to the paths list if it's not empty
    if current_path:
        paths.append(current_path)

# Convert paths to DataFrame
paths_df = pd.DataFrame(paths)


# Flatten all the paths into a single list of articles
articles_in_paths = paths_df.stack().unique()

# Map the articles to their IDs and handle missing keys
with open('articles_ids.tsv', 'w') as f:
    for article in articles_in_paths:
        article_id = article_to_id.get(article.strip())  # Use get to return None if not found
        if article_id is not None:
            f.write(f"{article_id}\t{article}\n")
        else:
            print(f"Article '{article}' not found in article_to_id mapping.")


# Extract unique transitions from the paths_df
transitions = set()

# Iterate through each row in paths_df (each row represents a path)
for _, row in paths_df.iterrows():
    path = row.dropna().values
    for fr, to in zip(path, path[1:]):
        transitions.add((fr, to))

# Convert the set of transitions to a DataFrame
links_df = pd.DataFrame(list(transitions), columns=['linkSource', 'linkTarget'])

# Save links_df as "links.tsv"
links_df.to_csv('links_new.tsv', sep='\t', index=False, header=False)


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



edge_ids = {}
for i, (s, r) in enumerate(zip(links_df.linkSource, links_df.linkTarget)):
    if s in article_to_id and r in article_to_id:
        edge_ids[(article_to_id[s], article_to_id[r])] = i

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



import torch

# Map the source and target articles to their IDs and convert to tensors
senders_tensor = torch.tensor(links_df.linkSource.map(get_article_id).tolist())
receivers_tensor = torch.tensor(links_df.linkTarget.map(get_article_id).tolist())


import torch

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda")

# Now you can send your tensors to the chosen device
senders = senders_tensor.to(device)
receivers = receivers_tensor.to(device)

# Concatenate tensors
concatenated_tensor = torch.cat([
    senders,
    receivers
    # ... other tensors you wish to concatenate
], dim=0)



graph = Graph(senders=senders, receivers=receivers, nodes=None, edges=None,
              n_node=len(articles_df), n_edge=len(links_df))

"""
Save Trajectories
"""
trajectory_articles = []

for _, row in paths_df.iterrows():
    trajectory = [article for article in row[:10] if pd.notnull(article)]
    trajectory_articles.append(trajectory)

for idx, traj in enumerate(trajectory_articles[:3]):  # Printing the first 3 for demonstration
    print(f"trajectory_articles[{idx}] ={{list:{len(traj)}}} {traj}")


# trajectory articles = (list:51312)
# trajectory articles[0] ={list:9} ['14th_century', '15th_century', '16th_century', 'Pacific_Ocean', 'Atlantic_Ocean', 'Accra', 'Africa', 'Atlantic_slave_trade', 'African_slave_trade']
# trajectory_articles[1] ={list:5} ['14th_century', 'Europe', 'Africa', 'Atlantic_slave_trade', 'African_slave_trade']
# trajectory_articles[2] ={list:8} ['14th_century', 'Niger', 'Nigeria', 'British_Empire', 'Slavery', 'Africa', 'Atlantic_slave_trade', 'African_slave_trade']


# save length
# for every path it saves its length, namely how many links there are in the path
# first value is the id and the second one the length of the path
# there are 51312 records in total
# 0	9
# 1	5
# 2	8
# 3	4
# 4	7
# 5	6

with open(os.path.join(output_dir, 'lengths2.txt'), 'w') as f:
    for i, articles in enumerate(trajectory_articles):
        f.write("{}\t{}\n".format(i, len(articles)-1))  # Subtract 1 for the integer.


# save observations
# for each path, it saves the article's id
# In the first line there are the total number of observations
# 305210	1
# 10	1.0
# 12	1.0
# 15	1.0
# 3134	1.0
# 377	1.0
# 105	1.0
with open(os.path.join(output_dir, 'observations2.txt'), 'w') as f:
    print(list(map(len, trajectory_articles)))   # [9, 5, 8, 4, 7, 6, 4, 6, 4, 7, 11, 10, 5....]
    num_observations = sum(map(len, trajectory_articles))   # 305210
    f.write("{}\t{}\n".format(num_observations, 1))
    for articles in trajectory_articles:
        for article in articles[1:]:  # Skip the first element which is the index
            f.write("{}\t{}\n".format(article_to_id[article], 1.))

# save paths
# computed only on the train_dataset
# it saves the id of traversed edges
traversed_edge_count = np.zeros(graph.n_edge)
print(graph.n_edge)
print(len(links_df))

test_dataset_start = int(len(trajectory_articles) * 0.8)
print("edge counts only until index {}/{}".format(test_dataset_start, len(trajectory_articles)))

missing_transitions = []  # Store missing transitions for further investigation.

with open(os.path.join(output_dir, 'paths2.txt'), 'w') as f:
    num_paths = sum(map(len, trajectory_articles)) - len(trajectory_articles)
    f.write("{}\t{}\n".format(num_paths, 1))

    for i, articles in enumerate(trajectory_articles):  # for each path
        for fr, to in zip(articles, articles[1:]):  # for each transition/link in a path
            try:
                edge = edge_ids[(article_to_id[fr], article_to_id[to])]
                if i < test_dataset_start:
                    traversed_edge_count[edge] += 1
            except KeyError as e:
                print('no link between {} -> {} for traj {}. Error: {}'.format(fr, to, i, e))
                edge = -1  # Assigning a special edge ID for missing transitions
                missing_transitions.append((fr, to))  # Add to the list for investigation.
            f.write("{}\n".format(edge))

# Optionally: Save missing transitions to a separate file for further investigation
with open(os.path.join(output_dir, 'missing_transitions.txt'), 'w') as f:
    for fr, to in missing_transitions:
        f.write("{} -> {}\n".format(fr, to))

"""
Extract categories
"""
with open("categories2.tsv", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Removing stray quotes
cleaned_lines = [line.replace('"', '') for line in lines]

with open("categories2_cleaned.tsv", "w", encoding="utf-8") as file:
    file.writelines(cleaned_lines)


categories_df = pd.read_csv('categories2_cleaned.tsv', sep='\t', header=None, names=['article', 'category'])
print(categories_df.head())
article_by_category = categories_df.drop_duplicates(subset='article').merge(articles_df, left_on='article', right_on='article', how='outer')
article_by_category.category[article_by_category.category == np.NaN] = ""
article_by_category.head(20)

def category_extractor(cat):
    if cat is np.NaN:
        return {}
    return {"category{}".format(i): c for i, c in enumerate(cat.split('.'))}


"""
Feature extraction - TF-IDF
"""

def process_filename(filename):
    # If the filename is 'Heraklion_(regional_unit).txt', do not alter it
    if filename == 'Heraklion_(regional_unit).txt':
        return filename
    # For other filenames, remove the specific characters
    else:
        for char in ['(', ')', '_', '/', '-', '.', '!',',']:
            filename = filename.replace(char, '')
        return filename

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# 1. Ensure output_dir is correct
print(output_dir)  # check its value


valid_articles = articles_df.article.dropna().str.strip().loc[lambda x: x != ""]

def find_text_file(article, base_dirs=['text', 'text2']):
    """
    Attempts to find the text file corresponding to an article within the specified base directories.

    :param article: The name of the article to find the text file for.
    :param base_dirs: List of base directories to search within.
    :return: The path to the text file if found, otherwise None.
    """
    for base_dir in base_dirs:
        # Generate the potential file path
        file_path = f"/media/data/marthass/wikispeedia-paths-dual-hypergraph-features-main/gretel/{base_dir}/{process_filename(article)}.txt"
        # Check if the file exists
        if os.path.exists(file_path):
            return file_path
    # Return None if the file does not exist in any of the base directories
    return None

def check_files_existence(articles_df):
    valid_articles = []
    missing_files = []
    
    for article in articles_df['article'].dropna().str.strip():
        file_path = find_text_file(article)
        if file_path:
            valid_articles.append(article)
        else:
            missing_files.append(article)
            
    return valid_articles, missing_files

valid_articles, missing_files = check_files_existence(articles_df)

if missing_files:
    print(f"Missing files for articles: {missing_files}")

missing_articles = check_files_existence(articles_df)


# Use a list comprehension along with the defined function to find all valid file paths
files = [find_text_file(article) for article in valid_articles]

# Filter out any None values if the file was not found in either directory
files = [file for file in files if file is not None]

article in valid_articles

if len(files) != len(valid_articles):
    print(f"Warning: Number of found files ({len(files)}) does not match number of articles ({len(valid_articles)}).")


# Assume article_to_id is a dictionary mapping article names to node IDs
valid_article_ids = [article_to_id[article] for article in valid_articles]

# Now you have valid_article_ids which should be used for filtering edges
valid_article_ids_set = set(valid_article_ids)


# Create TF-IDF vectors only for articles with corresponding text files
tf_idf = TfidfVectorizer(input='filename', stop_words='english')
tfidf_vectors = tf_idf.fit_transform(files)
distances = cosine_similarity(tfidf_vectors)

# Assume graph is an object with 'senders' and 'receivers' attributes that are tensors of node indices
# Filter out edges that have senders and receivers in the valid_article_ids_set

# Calculate a mask for valid edges where both the sender and receiver are in the valid_article_ids_set

# Assuming `traversed_edge_feature` is a tensor with one feature per edge,
# and `graph.senders` and `graph.receivers` are tensors representing the edges.
# The `valid_edges_mask` should be created as follows:



valid_edges_mask = torch.tensor([
    (s.item() in valid_article_ids_set and r.item() in valid_article_ids_set)
    for s, r in zip(graph.senders, graph.receivers)
])



filtered_senders = graph.senders[valid_edges_mask]
filtered_receivers = graph.receivers[valid_edges_mask]

# Update the graph's senders and receivers
graph.senders = filtered_senders
graph.receivers = filtered_receivers


distances = cosine_similarity(tfidf_vectors)

plt.imshow(distances)
plt.colorbar()

valid_indices = set(range(912))  # Assuming you have 7307 nodes
invalid_indices = set(article_to_id.values()) - valid_indices
print("Invalid Node Indices:", invalid_indices)

# After creating TF-IDF vectors...

# Create a mapping from old node IDs to new TF-IDF indices
node_id_to_tfidf_index = {node_id: index for index, node_id in enumerate(valid_article_ids)}

# Update the graph's senders and receivers using the new mapping
graph.senders = torch.tensor([node_id_to_tfidf_index[s.item()] for s in filtered_senders
                              if s.item() in node_id_to_tfidf_index], dtype=torch.long)
graph.receivers = torch.tensor([node_id_to_tfidf_index[r.item()] for r in filtered_receivers
                                if r.item() in node_id_to_tfidf_index], dtype=torch.long)

# Now, use the updated senders and receivers tensors to index into the TF-IDF distances matrix
# Ensure that your TF-IDF distance matrix is on the same device as your graph tensors if using CUDA
tfidf_distances = torch.tensor(distances, device='cuda:0').float()

# Use only the valid edges for which we have TF-IDF vectors
valid_edges_mask = (graph.senders < len(valid_article_ids)) & (graph.receivers < len(valid_article_ids))

# Index the TF-IDF distances matrix using the valid sender and receiver indices
tfidf_edge_distances = tfidf_distances[graph.senders[valid_edges_mask], graph.receivers[valid_edges_mask]]

# Apply mask to ensure only edges with valid sender and receiver indices are used
valid_edges_mask = (graph.senders < len(valid_article_ids)) & (graph.receivers < len(valid_article_ids))

# Use the mask to filter the senders and receivers
filtered_senders = graph.senders[valid_edges_mask]
filtered_receivers = graph.receivers[valid_edges_mask]

# Now index the tfidf_distances with the filtered senders and receivers
tfidf_edge_distances = tfidf_distances[filtered_senders, filtered_receivers]




print("Senders tensor shape:", graph.senders.shape)
print("Receivers tensor shape:", graph.receivers.shape)

missing_files = [f for f in files if not os.path.exists(f)]
if missing_files:
    print("Missing Files:", missing_files)
    # Removing missing files from the list
    files = [f for f in files if f not in missing_files]

print("Size of graph.senders:", len(graph.senders))
print("Size of graph.receivers:", len(graph.receivers))
print("Size of filtered_senders:", len(filtered_senders))
print("Number of valid articles:", len(valid_article_ids))
print("Size of filtered_receivers:", len(filtered_receivers))

# The number of sender/receiver pairs should be the same
assert graph.senders.shape == graph.receivers.shape

# The shape of the TF-IDF distances matrix should be (number of valid articles, number of valid articles)
assert distances.shape == (len(valid_article_ids), len(valid_article_ids))

# Example debugging output if assertions fail
if graph.senders.shape[0] != graph.receivers.shape[0]:
    print(f"Mismatch between senders and receivers counts: {graph.senders.shape[0]} vs {graph.receivers.shape[0]}")

if distances.shape[0] != len(valid_article_ids):
    print(f"Mismatch in distances matrix size vs number of valid articles: {distances.shape[0]} vs {len(valid_article_ids)}")







def closest_articles(article, distances, k=5):
    id_ = article_to_id[article]
    distance_others = distances[id_]
    sorted_indices = np.argsort(distance_others)[::-1]
    sorted_indices = sorted_indices[sorted_indices != id_]
    top_ids = sorted_indices[:k]
    return list(zip(map(id_to_article.__getitem__, top_ids), distances[id_][top_ids]))


closest_articles('Central_Macedonia', distances)


"""
Fast text embeddings
"""

with open(os.path.join(raw_input_dir, 'article_embeddings.txt'), 'r') as f:
    line_count = sum(1 for _ in f)

print(f"The file has {line_count} lines.")

if not graph.receivers.numel() or graph.senders.shape[0] != graph.receivers.shape[0]:
    raise ValueError("Either receivers tensor is empty or senders and receivers tensors have mismatched shapes!")


# Read the number of lines in articles.tsv to determine the number of articles
with open(os.path.join(raw_input_dir, 'articles.tsv'), 'r') as articles_file:
    n_articles = 912

# Read the article embeddings and set the array size dynamically
with open(os.path.join(raw_input_dir, 'article_embeddings.txt'), 'r') as f:
    first_line = f.readline().strip()  # Read the first line
    d = len(first_line.split())  # Determine the number of dimensions based on the first line
    f.seek(0)  # Reset the file pointer to the beginning of the file

    # Initialize the embedding matrix with zeros, now using the actual number of articles
    emb = np.zeros([n_articles, d])

    for i, line in enumerate(f):
        values = [float(x) for x in line.split()]
        if len(values) != d:
            raise ValueError(f"Line {i + 1} in the embeddings file has a different dimension than expected!")
        if i < n_articles:  # Ensure we don't go out of bounds
            emb[i] = values
        else:
            break  # If there are more lines in the embeddings file than articles, stop readin



emb_distances = cosine_similarity(emb)

plt.imshow(emb_distances)
plt.colorbar()
closest_articles("Central_Macedonia", emb_distances)

emb_distances_norm = emb_distances - emb_distances.min()
emb_distances_norm /= emb_distances_norm.max()
plt.imshow(emb_distances_norm)
plt.colorbar()


"""
Add features to graph
"""

# NODE FEATURES
embeddings = torch.tensor(emb).float()
embeddings_norm = embeddings.norm(dim=1)
# out_d = graph.out_degree_counts.float()
# in_d = graph.in_degree_counts.float()

out_d = torch.bincount(graph.senders.long()).float()
#in_d = torch.bincount(graph.receivers.long()).float()

# Ensure embeddings are aligned
aligned_embeddings = torch.zeros((graph.n_node, 300))
for id_ in range(min(len(emb), graph.n_node)):
    aligned_embeddings[id_] = torch.tensor(emb[id_])

# Ensure that the receivers tensor has valid data
if len(graph.receivers) > 0 and graph.receivers.max() > 0:
    in_d = torch.bincount(graph.receivers)

    print(f"Size of in_d tensor: {in_d.size()}")

    # NODE FEATURES
    embeddings_norm = aligned_embeddings.norm(dim=1)
    out_d_normalized = out_d.unsqueeze(1) / out_d.max()
    in_d_normalized = in_d.unsqueeze(1) / in_d.max()

    graph.nodes = torch.cat([
        out_d_normalized,
        in_d_normalized,
        # Add other tensors here if necessary
    ], dim=0)
else:
    print("Warning: Receivers tensor is empty or contains invalid data!")




# EDGE FEATURES


# Assuming emb_distances_norm and distances are computed correctly
fasttext_distances = torch.tensor(emb_distances_norm).float()
tfidf_distances = torch.tensor(distances).float()

# Normalize traversed_edge_feature
traversed_edge_feature = torch.tensor(1. * traversed_edge_count / traversed_edge_count.max()).float()

# Apply the mask to both tfidf_distances and traversed_edge_feature
# Make sure tfidf_distances is also indexed by edges and not by nodes

# First, print out the sizes to understand the mismatch.
print("graph.senders.size(0):", graph.senders.size(0))
print("graph.receivers.size(0):", graph.receivers.size(0))
print("traversed_edge_feature.size(0):", traversed_edge_feature.size(0))

# Then check that all the sizes match, as before.
assert graph.senders.size(0) == graph.receivers.size(0) == traversed_edge_feature.size(0), \
       "Sizes of senders, receivers, and features must match."


assert valid_edges_mask.size(0) == traversed_edge_feature.size(0), \
       "The size of valid_edges_mask must match the number of edges in traversed_edge_feature."

# Assuming tfidf_edge_distances is correctly filtered as well:
tfidf_edge_distances = tfidf_edge_distances[valid_edges_mask]
traversed_edge_feature = traversed_edge_feature[valid_edges_mask]

print("Max sender index:", graph.senders.max())
print("Max receiver index:", graph.receivers.max())
print("TF-IDF distances tensor shape:", tfidf_edge_distances.shape)
print("Traversed edge feature tensor shape:", traversed_edge_feature.shape)

tfidf_edge_distances = tfidf_edge_distances.to(device)
traversed_edge_feature = traversed_edge_feature.to(device)

# Now you can stack them because they have the same size
graph.edges = torch.stack([
    tfidf_edge_distances,
    traversed_edge_feature
], dim=1)





# Ensure no indices are out of bounds
assert graph.senders.max() < tfidf_distances.size(0), "Sender index out of bounds"
assert graph.receivers.max() < tfidf_distances.size(1), "Receiver index out of bounds"


print(len(graph.receivers))
print(graph.receivers)

if len(graph.receivers) == 0:
    graph.receivers = torch.zeros_like(graph.senders)



graph.write_to_directory(output_dir)

"""
Given as target 
"""
targets = [t[-1] for t in trajectory_articles]
given_as_target = torch.zeros(graph.n_node, dtype=torch.uint8)
for target in targets:
    given_as_target[article_to_id[target]] = 1

torch.save(given_as_target, os.path.join(output_dir, "given_as_target.pt"))


"""
Node target preprocessing
"""


pairwise_features = torch.cat([
    fasttext_distances.unsqueeze(-1),
    tfidf_distances.unsqueeze(-1)
], dim=-1)

torch.save(pairwise_features, os.path.join(output_dir, "pairwise_node_features.pt"))

"""
Siblings nodes
"""
all_categories = sorted(list(set(categories_df.category)))
category_to_id = {c: i for i, c in enumerate(all_categories)}

# -- n_article, n_cat
article_category_one_hot = torch.zeros([len(articles_df), len(all_categories)], dtype=torch.uint8)
for i, row in categories_df.iterrows():
    article_category_one_hot[article_to_id[row.article], category_to_id[row.category]] = 1

siblings = torch.einsum("ac,bc->ab", [article_category_one_hot.float(), article_category_one_hot.float()])
siblings += torch.eye(siblings.shape[0])
siblings = siblings > 0

torch.save(siblings, os.path.join(output_dir, "siblings.pt"))


"""
Export to .gml
"""
graph = nx.DiGraph()
graph.add_nodes_from([(row.article, {'id': i, 'article': row.article, **category_extractor(row.category)}) for i, (_, row) in enumerate(article_by_category.iterrows())])
article_to_id_f = article_to_id.__getitem__
graph.add_edges_from(zip(links_df.linkSource, links_df.linkTarget, map(lambda x: {'Weigth': int(x)}, traversed_edge_count)))

# Sample some path
num_samples = 20
sampled_indices = np.random.choice(len(trajectory_articles), num_samples, replace=False)
sampled_paths = [trajectory_articles[i] for i in sampled_indices]

# for i in range(len(sampled_paths)):
#     nx.set_node_attributes(graph, 0, "path{}".format(i))
for i, path in enumerate(sampled_paths):
    path_name = 'pathx{}x{}x{}'.format(len(path), path[0], path[-1])
    path_name = re.sub(r'\W+', '', path_name).replace("_", "")
    for step, node in enumerate(path):
        graph.nodes[node][path_name] = (step + 1) * 10

nx.write_gml(graph, os.path.join(output_dir, 'graph.gml'))


"""
Create dataframe with article's name and one hot vector for the first level category
example:
10th_century	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
11th_century	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
12th_century	[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"""
def get_multi_hot(s, col):
    if col == 1:
        return [1 if c in s else 0 for c in cats_1]
    else:
        print("Wrong argument provided")


categories_df = pd.read_csv(os.path.join(raw_input_dir, 'categories2.tsv'),
                            sep='\t',
                            comment='#',
                            header=None,
                            names=['article', 'category'])
# categories_df.shape = (5204, 2)  # some articles belong to more than one categories
categories_splitted_df = categories_df['category'].str.split('.', expand=True)      # dataframe only with the splitted categories
articles_categories_df = categories_splitted_df.groupby(categories_df['article']).agg(lambda x: set(x))    # new_df.shape = (4598, 4)

cats_1 = categories_splitted_df[1].unique()
cats_1 = [a for a in cats_1 if a is not None]


articles_categories_df[1] = articles_categories_df[1].apply(get_multi_hot, args=[1])
articles_categories_df[0] = articles_categories_df[1]
del articles_categories_df[1]
del articles_categories_df[2]
del articles_categories_df[3]
articles_categories_df = articles_categories_df.rename(columns={0: 'embedding'})

first_category_embeddings = articles_df.merge(articles_categories_df, on=['article'], how='left', indicator=True)
# check for articles without categories - set one hot vector with zeros to all categories
first_category_embeddings.loc[first_category_embeddings['_merge'] == 'left_only', 'embedding'] = 0
first_category_embeddings['embedding'] = first_category_embeddings['embedding'].apply(lambda x: x if not isinstance(x, int) else [0 for _ in range(15)])
del first_category_embeddings['_merge']
first_category_embeddings.to_csv(os.path.join(raw_input_dir, "first_category_embeddings.tsv"), sep='\t', index=False, header=False)

