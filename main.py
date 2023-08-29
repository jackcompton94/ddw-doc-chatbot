from flask import Flask, request, jsonify
from flask_cors import CORS
from src import util, bot_functions
from src.crawlers import product_doc_crawler, ctk_doc_crawler
import openai
import os

# Get env variables from Heroku
DW_PORT = os.getenv("PORT", 5000)
DW_OPENAI_KEY = os.getenv("DW_OPENAI_KEY")

# Initialize the Flask app with CORS
app = Flask(__name__)
CORS(app)

# Load embeddings into DataFrame at runtime
embeddings_df = util.load_embeddings_to_df('data/csvs/embeddings.csv')


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    question = data.get('question')

    # Authorize the current user
    openai.api_key = DW_OPENAI_KEY

    # Get the bots response
    response = bot_functions.get_response(question, embeddings_df)

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=DW_PORT)
