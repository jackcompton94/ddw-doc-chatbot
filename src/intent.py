import openai


def get_intent(question):

    content = "You are an intent classifier bot that provides output to a support bot to help them understand nuanced queries." \
              "INSTRUCTIONS:\n" \
              "1. Examine the question carefully to provide the additional context needed." \
              "2. Determine the underlying intent of the users question" \
              "3. Respond in 3 words or less" \
              "4. Only use lowercase and spaces" \
              "5. Do NOT answer or address the question directly" \
              "CLASSIFICATION:"

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
    print(f"intent: {intent}")

    return intent
