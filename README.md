
# Wikipedia Central Macedonia Dataset

## Description

The Wikipedia Central Macedonia (WCM) dataset is an innovative dataset designed to support the advancement of graph-based models for understanding and predicting navigational patterns within complex networks. Specifically tailored for the Central Macedonia region in Greece, the WCM dataset plays a crucial role in facilitating path generation and extrapolation methods that predict Wikipedia navigation paths with higher efficiency.

Our approach involves a crawling process that mimics human navigation through Wikipedia, thereby creating a dataset that not only caters to the development of graph neural networks but also enhances models' effectiveness by integrating hypergraph features with graph edge features.


## Visual Representation of Graphs

### Dense Graph

![Dense Graph](images/dense_graph.png)

This image represents the dense graph, highlighting the closely knit connections achieved by selecting the next article from the first five links.

### Sparse Graph

![Sparse Graph](images/sparse.png)

This image shows the sparse graph, illustrating the effect of broadening the selection threshold, leading to a graph with less dense connections.



## Dataset Details

The WCM dataset comprises two distinct graph types: a dense graph and a sparse graph. The density of these graphs is determined during the crawling process by the selection mechanism for the next article to visit, as implemented in the `wikicrawl.py` file:

- **Dense Graph**: Achieved by setting a limit where the crawler selects the next article from the first five links encountered. This constraint (`if len(visited_articles) >= 5: break`) in `wikicrawl.py` ensures a denser connectivity among the articles, as it restricts the exploration to a narrower scope.
- **Sparse Graph**: Generated by allowing the crawler to select from a broader array of links. By increasing the threshold beyond five in `wikicrawl.py`, the crawler diversifies its path, leading to a sparser graph with less dense connections.

This distinction allows for comparative studies and analyses on how graph density affects navigational pattern predictions and model performance.


## Repository Contents

- `wikicrawl.py`: Generates a file containing 3,000 navigation routes starting from Central Macedonia.
- `article.py`: Isolates unique articles from the generated routes.
- `categories.py`: Determines the categories for each unique article.
- `wikipedia_preprocessing.py`: Produces auxiliary files such as `length.txt` and `observations.txt` and `links.tsv`.
- `nodes_edges.py`: Creates files detailing nodes and edges for graph representation.

## Installation Instructions

To use this dataset and the associated scripts, follow these steps:

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine.
3. Install the necessary Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the scripts in the order mentioned to generate the dataset and associated files.

## Usage

To generate the complete Wikipedia Central Macedonia dataset and its graph representations, execute the scripts in the following sequence:

1. **Data Crawling**:
   ```bash
   python wikicrawl.py
   ```
2. **Article Isolation**:
   ```bash
   python article.py
   ```
3. **Category Finding**:
   ```bash
   python categories.py
   ```
4. **Preprocessing**:
   ```bash
   python wikipedia_preprocessing.py
   ```
5. **Nodes and Edges Creation**:
   ```bash
   python nodes_edges.py
   ```
## Acknowledgment

-Path Inference with Dual Hypergraph Features [[code]](https://github.com/jbcdnr/gretel-path-extrapolation)- 
[[paper]](https://ieeexplore.ieee.org/document/10191161)
