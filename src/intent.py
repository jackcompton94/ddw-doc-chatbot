import openai


def get_intent(question):

    content = "Please provide the explicit intent for the user query below and only respond with the intent that you classify it to be:"

    # Generate response from OpenAI
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": question},
        ],
        max_tokens=500,
        temperature=0.5
    )

    print(response.choices[0].message['content'])

    # Extract the assistant's reply
    return response.choices[0].message['content']
