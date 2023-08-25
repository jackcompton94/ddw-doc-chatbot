import pandas as pd
import openai
from ast import literal_eval


# Gets the embedding of a text string
def get_embedding(text):
    model = "text-embedding-ada-002"
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


# Load CSV and convert embedding columns to list objects
def load_embeddings_to_df(embeddings_file):
    embeddings_df = pd.read_csv(embeddings_file)

    # Convert string lists into Python list objects
    embeddings_df['title_embedding'] = embeddings_df['title_embedding'].apply(literal_eval)
    embeddings_df['content_embedding'] = embeddings_df['content_embedding'].apply(literal_eval)

    return embeddings_df


# Replace data.world specific terms/context to help tune the prompt engine
def preprocess_question(question):
    question = question.replace("collector", "metadata collector")
    question = question.replace("ctk", "catalog toolkit")

    if "lineage" in question and "manta" not in question:
        question = question.replace("lineage", "explorer lineage")

    return question
