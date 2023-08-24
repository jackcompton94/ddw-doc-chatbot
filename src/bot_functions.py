import openai
from src import util
from src import intent_generator
from src import response_generator


def get_response(question, embeddings_df):
    try:
        # Preprocess users question to replace data.world specific shorthand with documented terms
        question = util.preprocess_question(question)

        # Get the intent from the user's query
        intention = intent_generator.get_intent(question)

        # Get the embedding for the user's question and intention
        embed_question = util.get_embedding(question)
        embed_intention = util.get_embedding(intention)

        # Generate the appropriate prompt for the user's question
        prompt = response_generator.get_best_response(question, intention, embeddings_df, embed_question, embed_intention)

        # Generate response from OpenAI
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=2000,
            temperature=0
        )

        # Return the generated response
        return response.choices[0].text.strip()

    except openai.error.OpenAIError as e:
        error_message = "Invalid token or an error occurred while communicating with OpenAI."
        return error_message
