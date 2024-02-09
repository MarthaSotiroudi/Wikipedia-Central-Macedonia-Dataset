import random
import requests
from bs4 import BeautifulSoup
import csv
import time

# Define the delay duration (in seconds) between requests
delay_duration = 1.0


def extract_internal_links_with_delay(url):
    time.sleep(delay_duration)
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find(id="content")

        internal_links = []
        for link in content.find_all("a", href=True):
            href = link.get("href")
            if href.startswith("/wiki/"):
                title = href[len("/wiki/"):]
                internal_links.append(title)
        return internal_links
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return []


def is_valid_title(title):
    excluded_terms = ["Talk", "User", "File", "ISO", "_(identifier)", "%", "#", ":"]
    return not any(term in title for term in excluded_terms) and not title.isdigit()


num_routes = 3000
routes = []
unique_articles = set()

for route_num in range(1, num_routes + 1):
    while True:
        route_length = random.randint(5, 7)
        start_article_title = "Central_Macedonia"
        visited_articles = [start_article_title]

        for _ in range(route_length - 1):
            current_article = visited_articles[-1]
            next_article_url = f"https://en.wikipedia.org/wiki/{current_article}"
            external_links = extract_internal_links_with_delay(next_article_url)

            # Filter external links
            external_links = [link for link in external_links if is_valid_title(link) and link not in visited_articles][:5]

            if external_links:
                next_article = random.choice(external_links)
                visited_articles.append(next_article)
            else:
                break  # Break if no valid external links are found

        if len(visited_articles) >= 5:
            break

    routes.append(visited_articles)
    unique_articles.update(visited_articles)

    if route_num % 100 == 0:
        with open('generated_routes2.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for route in routes:
                csvwriter.writerow(route)
        routes = []  # Reset routes list

    print(f"Route {route_num}:", visited_articles)

# Save the remaining routes
with open('generated_routes2.csv', 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for route in routes:
        csvwriter.writerow(route)

print("Routes saved to generated_routes.csv")
print(f"Total unique articles in all routes: {len(unique_articles)}")
