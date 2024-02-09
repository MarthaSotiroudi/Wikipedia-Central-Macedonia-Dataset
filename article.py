import csv

# List to store the filtered routes
filtered_routes = []

# Read the CSV file and extract routes
csv_file_path = '/media/data/marthass/wikispeedia-paths-dual-hypergraph-features-main/gretel/generated_routes.csv'
with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for i, row in enumerate(csv_reader):
        # Check if the row number is within the desired range and starts with 'Central_Macedonia'
        if 6001 <= i < 9001 and row[0] == 'Central_Macedonia':
            filtered_routes.append(row)
            # Break if we have collected 3000 such routes
            if len(filtered_routes) >= 3000:
                break

# Save the filtered routes to a TSV file
tsv_file_path_filtered = 'filtered_routes.tsv'
with open(tsv_file_path_filtered, 'w', newline='', encoding='utf-8') as tsvfile:
    tsv_writer = csv.writer(tsvfile, delimiter='\t')
    for route in filtered_routes:
        tsv_writer.writerow(route)

# Set to store unique articles from the filtered routes
unique_articles = set()
for route in filtered_routes:
    for article in route:
        if article.strip():
            unique_articles.add(article.strip())

# Save the unique articles from the filtered routes to a CSV file
csv_file_path_articles = 'unique_articles_filtered.csv'
with open(csv_file_path_articles, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    for article in unique_articles:
        csv_writer.writerow([article])

print(f"Saved filtered routes to {tsv_file_path_filtered}")
print(f"Saved {len(unique_articles)} unique articles to {csv_file_path_articles}")






