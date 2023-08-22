import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_max_similarity(embed_question, embed_intention, embeddings_df):
    # Initialize a list to store similarity scores
    similarities = []

    # Iterate over each row in the DataFrame
    for idx, row in embeddings_df.iterrows():
        # Get the embeddings of the current title and content
        embedding_title = np.array(row['title_embedding'])
        embedding_content = np.array(row['content_embedding'])

        # Calculate the cosine similarity between user's question and current title/content embedding
        similarity_title = cosine_similarity([embed_question], [embedding_title])[0][0]
        similarity_content = cosine_similarity([embed_question], [embedding_content])[0][0]

        # Calculate cosine similarity between intention embedding and title/content embedding
        similarity_intention_title = cosine_similarity([embed_intention], [embedding_title])[0][0]
        similarity_intention_content = cosine_similarity([embed_intention], [embedding_content])[0][0]

        # Weights to tune the importance of either the title, content, or intention (total = 1.0)
        title_weight = 0.3
        content_weight = 0.5

        # Normalize the intention similarity scores
        normalized_similarity_intention_title = (similarity_intention_title + 1) / 2
        normalized_similarity_intention_content = (similarity_intention_content + 1) / 2

        # Normalize the combined intention similarity
        normalized_combined_intention = (normalized_similarity_intention_title + normalized_similarity_intention_content) / 2

        # Weights for normalized intention similarity
        intention_weight = 0.2 * normalized_combined_intention

        # Combine the similarity scores
        combined_similarity = (
                title_weight * similarity_title +
                content_weight * similarity_content +
                intention_weight * normalized_combined_intention
        )

        # Append the combined similarity score to the list
        similarities.append(combined_similarity)

    # Find the index of the title with the highest similarity score
    best_idx = np.argmax(similarities)
    max_similarity = max(similarities)

    # TODO: Log to console the max similarity score
    print(f"max similarity: {max_similarity}")

    return best_idx, max_similarity
