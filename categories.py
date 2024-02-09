import csv
import time
from urllib.parse import quote
from SPARQLWrapper import SPARQLWrapper, JSON


def load_mapping(file_path):
    mapping = {}
    unique_subjects = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=':')
        for line_num, row in enumerate(reader, 1):
            if len(row) < 2:
                print(f"Warning: Malformed line at {file_path}:{line_num}: {row}")
                continue
            key = row[0].replace("'", "").strip()
            value = row[1].replace("'", "").strip()
            mapping[key] = value
            unique_subjects.add(value)

    return mapping, list(unique_subjects)


def determine_category(article, mapping, priorities):
    if article in mapping:
        return mapping[article]

    # If the article title contains the word "history"
    if "history" in article.lower():
        return "subject.History"

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    try:
        sparql.setQuery(f"""
            SELECT ?type WHERE {{
                <http://dbpedia.org/resource/{quote(article)}> rdf:type ?type .
            }}
        """)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        if "results" not in results or "bindings" not in results["results"]:
            return "subject.General"

        types = [result["type"]["value"] for result in results["results"]["bindings"]]

        # Using the dynamic priority list based on the mapping
        for value in priorities:
            if any(value in t for t in types):
                return value

        # Filter out undesired categories
        filters = ["wiki", "yago", "owl", "#"]
        specific_types = [t for t in types if not any(f in t.lower() for f in filters) and not t.endswith('Thing')]

        # Convert the category to the desired format
        if specific_types:
            category = specific_types[0].split("/")[-1].replace("_", ".")

            # Change "Person" into "People"
            if category == "Person":
                category = "People"

            return "subject." + category
    except Exception as e:
        print(f"Error while querying DBpedia for article: {article}. Error: {e}")
        return "subject.Error"

    return "subject.General"


if __name__ == '__main__':
    articles = open("articles.tsv", "r").readlines()
    mapping, priorities = load_mapping("mapping.csv")

    with open("categories2.tsv", "w", encoding='utf-8') as out:
        for article in articles:
            article = article.strip()
            category = determine_category(article, mapping, priorities)
            print(f"Processing article: {article} | Category: {category}")
            if category and category != "Not Found":
                out.write(f"{article}\t{category}\n")
            time.sleep(1)
