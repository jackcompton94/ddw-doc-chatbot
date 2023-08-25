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
    updated_data = []  # To store updated and new rows

    # Read existing titles and data from the CSV file
    with open(csv_file_path, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            existing_titles.add(row["title"])
            updated_data.append(row)

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
                # Update the corresponding row with new embeddings
                for existing_row in updated_data:
                    if existing_row["title"] == title:
                        existing_row["title_embedding"] = title_embedding
                        existing_row["content_embedding"] = content_embedding
                        print("Updating embeddings for:", title)
                        break
            else:
                # Add a new row with embeddings
                updated_data.append({
                    "title": title,
                    "content": content,
                    "url": url,
                    "title_embedding": title_embedding,
                    "content_embedding": content_embedding
                })
                print("Saving new embeddings:", title)

            # Add the new title to the set of existing titles
            existing_titles.add(title)

    # Write the updated and new rows back to the CSV file
    with open(csv_file_path, mode="w", newline="") as file:
        header = ["title", "content", "url", "title_embedding", "content_embedding"]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(updated_data)


# Adds pages in real-time to the JSON (this is just for monitoring)
def add_page(page, json_file_path):
    # Read existing JSON data from the file
    existing_data = []
    try:
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        # Handle if the file is empty or not valid JSON
        pass

    # Append the new JSON object to the existing data
    existing_data.append(page)

    # Write the updated data back to the file
    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=2)


# Replace data.world specific terms/context to help tune the prompt engine
def preprocess_question(question):
    question = question.replace("collector", "metadata collector")
    question = question.replace("ctk", "catalog toolkit")

    if "lineage" in question and "manta" not in question:
        question = question.replace("lineage", "explorer lineage")

    return question
