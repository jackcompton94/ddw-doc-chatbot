import openai
import string


def get_intent(question):
    content = (
        "You are an intent classifier trained to assist a support bot by generating 3 word (or less) intents that guide it to relevant documentation." 
        "\nFor example, 'Hello' == 'greeting', 'What is a hammer' == 'hammer definition', 'How do I use a hammer' == 'hammer guide', 'Send me hammer documentation' == 'hammer documentation'"
        "\nIf you cannot determine an intent, use 'unclear'"
        "\nBe concise as possible, yet capture the main idea of the user's query. Consider how your intent could be used by the support bot to locate accurate documentation."
    )

    # Generate response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": question},
        ],
        max_tokens=60,
        temperature=0
    )

    intent = response.choices[0].message['content'].translate(str.maketrans('', '', string.punctuation))

    # TODO: Log intent and move to dataset for persistence
    print(f"intent: {intent.lower()}")

    return intent
