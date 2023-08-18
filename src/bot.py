import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src import util


QUESTION_KEYWORDS = ["how", "what", "why", "where", "when",
                     "explain", "describe", "define", "clarify",
                     "understand", "learn", "tell me about",
                     "details on", "overview of"]

COMMAND_KEYWORDS = ["can", "send", "give", "show", "display",
                    "list", "provide", "share", "demonstrate",
                    "walk me through", "guide", "help with",
                    "perform", "execute"]


def generate_prompt(question, max_similarity, best_title, best_response, best_url):
    confidence_threshold = 0.74

    # Check if AI is confident in its response
    if max_similarity < confidence_threshold:

        # Low confidence suggests to check the docs directly and reach out to the support team
        return f"INSTRUCTIONS: You are a support bot for data.world. You are unsure of the answer, refer users to our support team for personalized assistance: https://support.data.world\n" \
               f"QUESTION: {question}\n" \
               f"RESPONSE:"

    # Check if the user's input contains keywords indicating a question
    elif any(keyword in question.lower() for keyword in QUESTION_KEYWORDS):
        return \
                f"DOCUMENTATION: {best_title} {best_response}\n URL:{best_url}\n" \
                f"INSTRUCTIONS: You are a support bot for data.world. Provide concise troubleshooting advice and key steps to address the issue the user is facing in their question below. Support your response with relevant information from the documentation above and provide the URL for reference.\n" \
                f"QUESTION: {question}\n " \
                f"RESPONSE:"

    # Check if the user's input contains keywords indicating a command
    elif any(keyword in question.lower() for keyword in COMMAND_KEYWORDS):
        return \
                f"DOCUMENTATION: {best_title} {best_response}\n URL:{best_url}\n" \
                f"INSTRUCTIONS: You are a support bot for data.world. Engage in a friendly conversation and offer assistance by answering the question below. If the user explicitly asks about functionality, features, or how to perform certain tasks, provide guidance by using the documentation above to support your response and provide the URL for reference.\n" \
                f"QUESTION: {question}\n " \
                f"RESPONSE:"

    # Provide a conversational response
    else:
        return \
                f"DOCUMENTATION: {best_title} {best_response}\n URL:{best_url}\n" \
                f"INSTRUCTIONS: You are a support bot for data.world. Provide a helpful and engaging response, only use the documentation above if relevant.\n" \
                f"QUESTION: {question}\n " \
                f"RESPONSE:"


def get_best_response(question, embeddings_df, embed_question):
    # Initialize a list to store similarity scores
    similarities = []

    # Iterate over each row in the DataFrame
    for idx, row in embeddings_df.iterrows():
        # Get the embeddings of the current title and content
        embedding_title = np.array(row['title_embedding'])
        embedding_content = np.array(row['content_embedding'])

        # Calculate the cosine similarity between user's question and current title embedding
        similarity_title = cosine_similarity([embed_question], [embedding_title])[0][0]

        # Calculate the cosine similarity between user's question and current content embedding
        similarity_content = cosine_similarity([embed_question], [embedding_content])[0][0]

        # Weights can be adjusted to tune the importance of either the title or content
        title_weight = 0.3
        content_weight = 0.7

        # Combine the similarity scores
        combined_similarity = title_weight * similarity_title + content_weight * similarity_content

        # Append the combined similarity score to the list
        similarities.append(combined_similarity)

    # Find the index of the title with the highest similarity score
    best_idx = np.argmax(similarities)

    # Get the title and content with the highest similarity score
    best_response = embeddings_df.loc[best_idx, 'content']
    best_title = embeddings_df.loc[best_idx, 'title']
    best_url = embeddings_df.loc[best_idx, 'url']

    max_similarity = max(similarities)

    # Log to console the max similarity score (testing)
    print(max_similarity)

    return generate_prompt(question, max_similarity, best_title, best_response, best_url)


def get_response(question, embeddings_df):
    try:
        # Preprocess users question to replace data.world specific shorthand with documented terms
        question = util.preprocess_question(question)

        # Get the embedding for the user's question
        embed_question = util.get_embedding(question)

        # Generate the appropriate prompt for the user's question
        prompt = get_best_response(question, embeddings_df, embed_question)

        # Generate response from OpenAI
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=250,
            temperature=0
        )

        # Return the generated response
        return response.choices[0].text.strip()

    except openai.error.OpenAIError as e:
        error_message = "Invalid token or an error occurred while communicating with OpenAI."
        return error_message
