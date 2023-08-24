import openai
from src import util
from src import intent_generator
from src import response_generator

# Initialize an empty conversation
conversation = []


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

        # # Testing gpt 3.5
        # # Add bot's prompt to the conversation
        # conversation.append({"role": "system", "content": prompt})
        #
        # # Add user's input to the conversation
        # conversation.append({"role": "user", "content": question})
        #
        # # Generate response from OpenAI in a conversation context
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=conversation,
        #     max_tokens=250,
        #     temperature=0
        # )
        #
        # # Extract the assistant's response and append to conversation
        # assistant_response = response.choices[0].message['content']
        # conversation.append({"role": "assistant", "content": assistant_response})
        #
        # # Return the generated response
        # return assistant_response.strip()

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
