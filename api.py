from flask import Flask, request, jsonify
import ollama
import arabic_reshaper
from bidi.algorithm import get_display
from time import sleep
app = Flask(__name__)
# Add a test endpoint
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly'
    })

@app.route('/predict', methods=['POST'])
def extract_ingredients():
    # Récupérer le fichier image envoyé par Flutter
    image = request.files['image']
    print(f"Received image: {image.filename}")
    image.save(image.filename)  # Sauvegarder temporairement

    response = ollama.chat(
        # model='llama3.2-vision',
        model='gemma3:4b',
        options={'temperature': 0},
        messages=[{
            'role': 'user',
            'content': '''
                You are given an image of a food product label. Please extract the list of ingredients as they appear on the label in *all available languages*. 
                Preserve the order and format and return them as they are , do not change the language, and only return the text of the ingredient and the weight or description.
            ''',
            'images': [image.filename]
        }]
    )

    output = response['message']['content']

    sleep(180)
    print('EXTRACTED RAW TEXT:\n', output)
    try:
        output = get_display(arabic_reshaper.reshape(output))
    except Exception:
        pass

    return jsonify({'ingredients': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)