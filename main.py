from flask import Flask, request, jsonify
from flask_cors import CORS
from src import util
from src import bot_functions
from src import config
from src.crawlers import product_doc_crawler
import openai

# Initialize the Flask app with CORS
app = Flask(__name__)
CORS(app)

# ######################################## Product Documentation Crawler ########################################
# # TODO: Add the parent URL from which to begin crawl
# product_doc_url = 'https://docs.data.world/en/160693-using-hoots-and-bb-bots-for-data-ops.html'
#
# # TODO: Name a json_file to embed from and the embeddings csv to append to
# json_file_path = '.data/jsons/bb_bot_scrape.json'
# embeddings_csv_path = '.data/csvs/embeddings.csv'
#
# # Crawl, scrape, embed, and store a doc page locally
# product_doc_crawler.scrape_doc_page(product_doc_url, json_file_path, embeddings_csv_path)

# Load embeddings into DataFrame at runtime
embeddings_df = util.load_embeddings_to_df('data/csvs/embeddings.csv')


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    question = data.get('question')

    # Authorize the current user
    openai.api_key = config.OPENAI_KEY

    # Get the bots response
    response = bot_functions.get_response(question, embeddings_df)

    # TODO: move to a dataset for persistent data later on
    print(f"question: {question}")
    print(f"response: {response}")

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run()
