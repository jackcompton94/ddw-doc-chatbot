from flask import Flask, request, jsonify
from flask_cors import CORS
import util
import bot
import openai

# Initialize the Flask app with CORS
app = Flask(__name__)
CORS(app)

# # ONLY RUN THIS WHEN THE DOCS PAGE IS UPDATED
# # Crawl the product documentation site
# crawler.scrape_doc_page('https://docs.data.world/en/98527-product-documentation.html')

# # ONLY RUN THIS WHEN THE DOCS PAGE IS UPDATED
# # Embed a scraped vx_scrape.json file
# util.embed_docs('v5_scrape.json')

# Load embeddings into DataFrame at runtime
embeddings_df = util.load_embeddings_to_df('../csvs/v5_embeddings.csv')


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
    app.run()
