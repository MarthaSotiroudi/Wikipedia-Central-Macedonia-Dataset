import pandas as pd
import numpy as np
import networkx as nx
import fasttext
import os
import pickle

raw_input_dir = '../workspace/wikispeedia/raw_input_data/'
output_dir = '/media/data/marthass/wikispeedia-paths-dual-hypergraph-features-main/gretel/'
input_dir = '/media/data/marthass/wikispeedia-paths-dual-hypergraph-features-main/gretel/'

"""
Load data
"""
articles_df = pd.read_csv('articles.tsv', delimiter='\t', header=None, names=['article'])
articles_df['article'] = articles_df['article'].str.replace('"', '')
articles_df.drop_duplicates(subset='article', inplace=True)

# Create the mappings

article_to_id = {a: i for i, a in enumerate(articles_df['article'])}
id_to_article = {i: a for i, a in enumerate(articles_df['article'])}

# Now read the 3000.tsv file
# Initialize an empty list to store the paths
paths = []

# Open the 3000.tsv file and read line by line
with open('3000.tsv', 'r') as file:
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
links_df.to_csv('links2.tsv', sep='\t', index=False, header=False)


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



article_to_id = {a: i for i, a in enumerate(articles_df.article.values)}
id_to_article = {i: a for i, a in enumerate(articles_df.article.values)}


articles_df.drop_duplicates(subset='article', inplace=True)


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



n_nodes = len(article_to_id)
n_edges = len(edge_ids)

n_hypernodes = n_edges
n_hyperedges = n_nodes



senders_id = []
receivers_id = []
senders_name = []
receivers_name = []
original_edge_features = {}
with open(os.path.join(input_dir, 'edges.txt'), 'r') as f:
    first_line = f.readline()
    for i, line in enumerate(f.readlines()):
        line = line.strip().split("\t")
        edge_id, out_id, in_id, tfidf, nof = int(line[0]), int(line[1]), int(line[2]), float(line[3]), float(line[4])
        senders_id.append(out_id)
        receivers_id.append(in_id)
        senders_name.append(articles_df.iloc[out_id][0])
        receivers_name.append(articles_df.iloc[in_id][0])
        original_edge_features[edge_id] = {}
        original_edge_features[edge_id]['out_id'] = out_id
        original_edge_features[edge_id]['in_id'] = in_id
        original_edge_features[edge_id]['tfidf'] = tfidf
        original_edge_features[edge_id]['nof'] = nof


original_node_features = {}
with open(os.path.join(input_dir, "nodes.txt"), 'r') as f:
    first_line = f.readline()
    for i, line in enumerate(f.readlines()):
        feat = []
        line = line.strip().split("\t")
        node_id, out_d, in_d = int(line[0]), float(line[1]), float(line[2])
        node_label = articles_df.iloc[node_id][0]
        original_node_features[node_id] = {}
        original_node_features[node_id]['label'] = node_label
        original_node_features[node_id]['out_degree'] = out_d
        original_node_features[node_id]['in_degree'] = in_d



with open(os.path.join(raw_input_dir, 'article_embeddings2.txt'), 'r') as f:
    n_articles, d = 912, 300
    article_embeddings = np.zeros([n_articles, d])
    for i, line in enumerate(f.readlines()):
        values = [float(x) for x in line.split()]
        article_embeddings[i] = values




G = nx.read_gml(os.path.join(input_dir, 'graph.gml'))
pr = nx.pagerank(G)
pr_values = np.fromiter(pr.values(), dtype=float)
pr_values_norm = pr_values - min(pr_values)
pr_values_norm /= max(pr_values_norm)
n_links = G.number_of_edges()




