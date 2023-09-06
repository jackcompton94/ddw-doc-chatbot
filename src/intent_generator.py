import openai
import string


def get_intent(question):
    content = \
        f"""
        Classify the text below, delimited by three dashes (-), by its intent only. 
        For example, 'Hello' == 'greeting', 'What is a hammer' == 'hammer definition', 'How do I use a hammer' == 'hammer guide', 'Send me hammer documentation' == 'hammer documentation'
        If you cannot determine an intent, use 'unclear'
        ---
        {question}
        ---
        """

    # Generate response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": content}
        ],
        max_tokens=60,
        temperature=0
    )

    intent = response.choices[0].message['content'].translate(str.maketrans('', '', string.punctuation))

    return intent
