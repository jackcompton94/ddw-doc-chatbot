import openai


def get_intent(question):
    content = (
        "You are an intent classifier bot designed to analyze user questions and provide explicit intents for a support bot's assistance."
        "\n\nINSTRUCTIONS:\n"
        "- Do NOT answer or address the user's question directly.\n"
        "- Your goal is to determine the underlying intent of the user's query.\n"
        "- Respond in 3 words or less to describe the intent clearly.\n"
        "- Consider the context and use concise language.\n"
        "\n\nCLASSIFICATION:"
    )

    # Generate response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": question},
        ],
        max_tokens=500,
        temperature=0
    )

    intent = response.choices[0].message['content']

    # TODO: Log intent and move to dataset for persistence
    print(f"intent: {intent.lower()}")

    return intent
