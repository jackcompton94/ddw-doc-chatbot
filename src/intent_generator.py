import openai
import string


def get_intent(question):
    content = \
        f"""
        Generate in 3 words (or less), an intent from the text below, delimited by three dashes (-), that could guide one to finding relevant documentation.
        \nFor example, 'Hello' == 'greeting', 'What is a hammer' == 'hammer definition', 'How do I use a hammer' == 'hammer guide', 'Send me hammer documentation' == 'hammer documentation'
        \nIf you cannot determine an intent, use 'unclear'
        \nBe concise as possible, yet capture the main idea. Consider how your intent could be used by a support bot to locate accurate documentation.
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
