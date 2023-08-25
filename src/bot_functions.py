import openai
from src import util
from src import intent_generator
from src import prompt_generator

# Initialize an empty conversation
conversation = []
total_tokens = 0


def get_response(question, embeddings_df):
    global total_tokens
    try:
        # Preprocess users question to replace data.world specific shorthand with documented terms
        question = util.preprocess_question(question.lower())

        # Get the intent from the user's query
        intention = intent_generator.get_intent(question)

        # Get the embedding for the user's question and intention
        embed_question = util.get_embedding(question)
        embed_intention = util.get_embedding(intention)

        # Generate the appropriate prompt for the user's question
        prompt = prompt_generator.get_best_document(question, intention, embeddings_df, embed_question, embed_intention)

        # # Generate response from gpt 3.5
        # max_tokens = 4096
        #
        # # Initialize conversation history
        # conversation = [{"role": "system", "content": prompt}]
        #
        # # Add user's input to the conversation
        # conversation.append({"role": "user", "content": question})
        #
        # while True:
        #     # Generate response from OpenAI in a conversation context
        #     response = openai.ChatCompletion.create(
        #         model="gpt-3.5-turbo",
        #         messages=conversation,
        #         max_tokens=1024,
        #         temperature=0
        #     )
        #
        #     # Extract the assistant's response
        #     assistant_response = response.choices[0].message['content']
        #
        #     # Calculate token usage
        #     total_tokens = response['usage']['total_tokens']
        #     print("Tokens used:", total_tokens)
        #
        #     # Check if the response exceeds available tokens
        #     if total_tokens > max_tokens:
        #         # Remove oldest message(s) to free up tokens
        #         conversation.pop(1)  # Remove the oldest user message
        #         if len(conversation) > 2 and conversation[1]['role'] == "assistant":
        #             conversation.pop(1)  # Remove the oldest assistant message
        #     else:
        #         # Add assistant's response to conversation
        #         conversation.append({"role": "assistant", "content": assistant_response})
        #         break  # Exit the loop as the response is within token limits
        #
        # # Return the generated response
        # return assistant_response.strip()

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
        print(e)
        error_message = "Invalid token or an error occurred while communicating with OpenAI."
        return error_message
