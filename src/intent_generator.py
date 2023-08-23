import openai


def get_intent(question):
    content = (
        "You are an intent classifier bot designed to analyze user questions and provide explicit intents to assist a support bot."
        "\n\nINSTRUCTIONS:\n"
        "- Do NOT answer the user's question directly.\n"
        "- Respond in 3 words or less to describe the intent clearly.\n"
        "- Your goal is to generate a clear and succinct intent that encapsulates the main topic of the user query.\n"
        "- Focus on crafting an intent that would assist another bot in efficiently searching through documentation to find the answer.\n"
        "- Consider how the generated intent could serve as a guide for the support bot to locate the specific documentation that addresses the user query.\n"
        "\n\nRESPONSE:\n"
    )

    # Generate response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": question},
        ],
        max_tokens=50,
        temperature=0
    )

    intent = response.choices[0].message['content']

    # TODO: Log intent and move to dataset for persistence
    print(f"intent: {intent.lower()}")

    return intent
