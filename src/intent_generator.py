import openai
import string


def get_intent(question):
    content = (
        "You are an intent classifier bot trained to assist a support bot by generating intents that guide it to relevant documentation."
        "\n\nINSTRUCTIONS:\n"
        "- Focus on crafting a clear, succinct intent in 3 words or less that captures the main topic of the user query.\n"
        "- Consider specific examples to cover a variety of intents and topics.\n"
        "- The generated intent will serve as a guide for the support bot to locate documentation.\n"
        "- Emphasize the importance of identifying keywords that are likely to appear in documentation.\n"
        "\n\nRESPONSE:\n"
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
