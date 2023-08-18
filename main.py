from flask import Flask, request, jsonify
from flask_cors import CORS
from src import util
from src import bot
import openai
import os

port = int(os.environ.get("PORT", 5000))

# Initialize the Flask app with CORS
app = Flask(__name__)
CORS(app)

# Load embeddings into DataFrame at runtime
embeddings_df = util.load_embeddings_to_df('v5_embeddings.csv')


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    api_key = request.headers.get('Authorization')
    question = data.get('question')

    # Authorize the current user
    openai.api_key = api_key

    # Get the bots response
    response = bot.get_response(question, embeddings_df)

    # TODO: move to a dataset for persistent data later on
    print(f"question: {question}")
    print(f"response: {response}")

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)

