from src import similarity_calculator

QUESTION_KEYWORDS = ["how", "what", "why", "where", "when",
                     "explain", "describe", "define", "clarify",
                     "understand", "learn", "tell me about",
                     "details on", "overview of", "can", "do", "is there"]

COMMAND_KEYWORDS = ["send", "give", "show", "display",
                    "list", "provide", "share", "demonstrate",
                    "walk me through", "guide", "help with",
                    "perform", "execute"]

MISC_INTENTS = ["greeting", "introduction", "identification", "identity",
                "bot information", "bot identity", "about bot", "capabilities"]


def generate_prompt(question, intention, max_similarity, best_title, best_content, best_url):
    confidence_threshold = 0.775

    # Irrelevant intent will engage with the user and encourage a question about data.world
    if any(keyword in intention.lower() for keyword in MISC_INTENTS):
        return \
            f"INSTRUCTIONS: You are a support bot for data.world. Engage in a friendly conversation and respond to the user. Let them know that you are here to help them understand the platform.\n" \
            f"QUESTION: {question}\n" \
            f"RESPONSE:"

    # Check if AI is confident in its response
    elif max_similarity < confidence_threshold or intention.lower() == 'unclear':

        # Low confidence suggests to check the docs directly and reach out to the support team
        return f"INSTRUCTIONS: You are a support bot for data.world. Engage in a friendly conversation but for this particular question, recommend the user to reword their prompt or reach out to the support team for personalized assistance: https://support.data.world\n" \
               f"QUESTION: {question}\n" \
               f"RESPONSE:"

    # Check if the user's input contains keywords indicating a question
    elif any(keyword in question.lower() for keyword in QUESTION_KEYWORDS):
        return \
            f"DOCUMENTATION: {best_title} {best_content}\n URL:{best_url}\n" \
            f"INSTRUCTIONS: You are a support bot for data.world speaking with a user in a chatbox. Provide a comprehensive answer to the users question. Utilize the documentation above and deliver a thorough response, make sure to address all aspects of the user's query. Include relevant examples, details, and explanations from the provided documentation and provide the URL.\n" \
            f"INTENTION: {intention}\n QUESTION: {question}\n" \
            f"RESPONSE:"

    # Check if the user's input contains keywords indicating a command
    elif any(keyword in question.lower() for keyword in COMMAND_KEYWORDS):
        return \
            f"DOCUMENTATION: {best_title} {best_content}\n URL:{best_url}\n" \
            f"INSTRUCTIONS: You are a support bot for data.world. Your goal is to assist users effectively when they issue commands or request actions. Engage in a helpful conversation, provide guidance, and explain how to execute the requested action using the documentation provided and provide the URL.\n" \
            f"INTENTION: {intention}\n QUESTION: {question}\n" \
            f"RESPONSE:"

    # Provide a conversational response
    else:
        return \
            f"DOCUMENTATION: {best_title} {best_content}\n URL:{best_url}\n" \
            f"INSTRUCTIONS: You are a support bot for data.world. Your aim is to provide friendly and informative responses that promote a positive user experience. Engage in a conversational manner, incorporating insights from the provided documentation to enrich your answers and provide the URL.\n" \
            f"INTENTION: {intention}\n QUESTION: {question}\n" \
            f"RESPONSE:"


def get_best_document(question, intention, embeddings_df, embed_question, embed_intention):
    best_idx, max_similarity = similarity_calculator.calculate_max_similarity(embed_question, embed_intention, embeddings_df)

    # Get the title and content with the highest similarity score
    best_content = embeddings_df.loc[best_idx, 'content']
    best_title = embeddings_df.loc[best_idx, 'title']
    best_url = embeddings_df.loc[best_idx, 'url']

    return generate_prompt(question, intention, max_similarity, best_title, best_content, best_url)
