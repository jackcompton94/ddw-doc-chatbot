import csv
import json
from ast import literal_eval
from src import config
import openai
import pandas as pd


openai.api_key = config.OPENAI_KEY


# Loads a JSON file into memory
def load_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


# Loads an embeddings CSV and converts the embedding columns to list objects
def load_embeddings_to_df(embeddings_file):
    embeddings_df = pd.read_csv(embeddings_file)

    # Convert string lists into Python list objects
    embeddings_df['title_embedding'] = embeddings_df['title_embedding'].apply(literal_eval)
    embeddings_df['content_embedding'] = embeddings_df['content_embedding'].apply(literal_eval)

    return embeddings_df


# Gets the embedding of a text string
def get_embedding(text):
    model = "text-embedding-ada-002"
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


# Embeds an entire JSON file and loads to CSV
def embed_docs(context_data, csv_file_path):
    data = []
    max_tokens = 8191

    json_file = load_json(context_data)

    for row in json_file:
        title = row['title']
        content = row['content']
        url = row['url']

        # Checks if the Title or Content contains only spaces/newlines
        if title.strip() and content.strip():

            # Limit the length to fit token constraints
            title = title[:max_tokens]
            content = content[:max_tokens]

            # Get embeddings for each
            title_embedding = get_embedding(title)
            content_embedding = get_embedding(content)

            # Store data in list
            data.append({
                "title": title,
                "content": content,
                "url": url,
                "title_embedding": title_embedding,
                "content_embedding": content_embedding
            })
            print("saving embeddings:", {
                "title": title,
                "content": content,
                "url": url,
                "title_embedding": title_embedding,
                "content_embedding": content_embedding
            })

    # Save data to CSV
    with open(csv_file_path, mode="w", newline="") as file:
        header = ["title", "content", "url", "title_embedding", "content_embedding"]
        writer = csv.DictWriter(file, fieldnames=header)

        writer.writeheader()
        writer.writerows(data)


# Updates the embeddings CSV with the latest JSON scrape
def update_embeddings(context_data, csv_file_path):
    max_tokens = 8191

    existing_titles = set()  # To keep track of existing titles

    # Read existing titles from the CSV file
    with open(csv_file_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            existing_titles.add(row["title"])

    updated_data = []

    json_file = load_json(context_data)

    for row in json_file:
        title = row['title']
        content = row['content']
        url = row['url']

        if title.strip() and content.strip():
            title = title[:max_tokens]
            content = content[:max_tokens]

            title_embedding = get_embedding(title)
            content_embedding = get_embedding(content)

            if title in existing_titles:
                # Update the corresponding row with new data
                existing_titles.remove(title)  # Remove the old title
                updated_data.append({
                    "title": title,
                    "content": content,
                    "url": url,
                    "title_embedding": title_embedding,
                    "content_embedding": content_embedding
                })
                print("updating embeddings for:", title)
            else:
                updated_data.append({
                    "title": title,
                    "content": content,
                    "url": url,
                    "title_embedding": title_embedding,
                    "content_embedding": content_embedding
                })
                print("saving new embeddings:", title)

            # Add the new title to the set of existing titles
            existing_titles.add(title)

    # Rewrite the entire CSV file with updated data
    with open(csv_file_path, mode="w", newline="") as file:
        header = ["title", "content", "url", "title_embedding", "content_embedding"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(updated_data)


# Replace data.world specific terms/context to help tune the prompt engine
def preprocess_question(question):
    question = question.replace("collector", "metadata collector")

    if "lineage" in question and "manta" not in question:
        question = question.replace("lineage", "explorer lineage")

    return question
