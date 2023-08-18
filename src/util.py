import json
import openai
import csv
import pandas as pd
from ast import literal_eval


# Load json file into memory
def load_json(file):
    with open(file, 'r') as json_file:
        data = json.load(json_file)
    return data


# Load CSV and convert embedding columns to list objects
def load_embeddings_to_df(embeddings_file):
    embeddings_df = pd.read_csv(embeddings_file)

    # Convert string lists into Python list objects
    embeddings_df['title_embedding'] = embeddings_df['title_embedding'].apply(literal_eval)
    embeddings_df['content_embedding'] = embeddings_df['content_embedding'].apply(literal_eval)

    return embeddings_df


# Gets the embedding value of a text string
def get_embedding(text):
    model = "text-embedding-ada-002"
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


# Embeds the vX_scrape.json file and loads to CSV
def embed_docs(context_data):
    data = []
    max_tokens = 8191

    file = load_json(context_data)

    for row in file:
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
            print({
                "title": title,
                "content": content,
                "url": url,
                "title_embedding": title_embedding,
                "content_embedding": content_embedding
            })

    # Save data to CSV - CHANGE THE TITLE TO AVOID OVERWRITING YOUR CSV
    csv_file = "../v5_embeddings.csv"
    with open(csv_file, mode="w", newline="") as file:
        header = ["title", "content", "url", "title_embedding", "content_embedding"]
        writer = csv.DictWriter(file, fieldnames=header)

        writer.writeheader()
        writer.writerows(data)


# Replace data.world specific terms/context to help tune the prompt engine
def preprocess_question(question):

    question = question.replace("collector", "metadata collector")

    return question
