from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import pandas as pd
from fuzzywuzzy import fuzz
import base64
import io
import numpy as np
import time



# Configure Tesseract before importing pytesseract
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
import pytesseract
from PIL import Image

# Set Tesseract executable path 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialisation de Flask
app = Flask(__name__)
CORS(app)

# Création du dossier temporaire s'il n'existe pas
if not os.path.exists("temp"):
    os.makedirs("temp")

# Nettoyage du texte OCR
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ſ0-9,.\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Chargement du dataset d'allergènes
try:
    df = pd.read_csv("allergenes_colorants.csv")
    colonnes = ["nom_fr", "nom_ar", "nom_en", "sous_type_fr", "sous_type_en", "sous_type_ar", "code", "proteine"]
    keywords = []
    for col in colonnes:
        if col in df.columns:
            keywords.extend(df[col].dropna().astype(str).str.lower().tolist())
    keywords = list(set(keywords))
    print(f"Loaded {len(keywords)} keywords from allergens dataset")
except Exception as e:
    print(f"Error loading allergens dataset: {e}")
    keywords = []
    df = pd.DataFrame()  # DataFrame vide en cas d'erreur

# Détection fuzzy améliorée utilisant toute la base de données
def detect_allergens_enhanced(text, threshold=75):
    """Détection améliorée utilisant toute la base de données"""
    detected = []
    text_lower = text.lower()
    
    try:
        for index, row in df.iterrows():
            # Collecte tous les termes à vérifier pour cet allergène
            terms_to_check = []
            
            # Ajouter les noms dans différentes langues
            for col in ['nom_fr', 'nom_en', 'nom_ar', 'sous_type_fr', 'sous_type_en', 'sous_type_ar', 'code']:
                if col in df.columns and pd.notna(row[col]):
                    term = str(row[col]).strip()
                    if term and term.lower() != 'nan':
                        terms_to_check.append(term)
            
            # Vérifier chaque terme
            for term in terms_to_check:
                # Recherche exacte d'abord
                if term.lower() in text_lower:
                    detected.append({
                        'term': term,
                        'nom_fr': str(row.get('nom_fr', '')),
                        'nom_en': str(row.get('nom_en', '')),
                        'nom_ar': str(row.get('nom_ar', '')),
                        'type': str(row.get('type', 'Allergène')),
                        'sous_type_fr': str(row.get('sous_type_fr', '')),
                        'sous_type_en': str(row.get('sous_type_en', '')),
                        'sous_type_ar': str(row.get('sous_type_ar', '')),
                        'code': str(row.get('code', '')),
                        'proteine': str(row.get('proteine', '')),
                        'match_type': 'exact'
                    })
                    break
                
                # Recherche fuzzy si pas de correspondance exacte
                else:
                    for word in text_lower.split():
                        if fuzz.ratio(term.lower(), word) >= threshold:
                            detected.append({
                                'term': term,
                                'nom_fr': str(row.get('nom_fr', '')),
                                'nom_en': str(row.get('nom_en', '')),
                                'nom_ar': str(row.get('nom_ar', '')),
                                'type': str(row.get('type', 'Allergène')),
                                'sous_type_fr': str(row.get('sous_type_fr', '')),
                                'sous_type_en': str(row.get('sous_type_en', '')),
                                'sous_type_ar': str(row.get('sous_type_ar', '')),
                                'code': str(row.get('code', '')),
                                'proteine': str(row.get('proteine', '')),
                                'match_type': 'fuzzy',
                                'confidence': fuzz.ratio(term.lower(), word)
                            })
                            break
                    
                    if detected and detected[-1]['term'] == term:
                        break
    
    except Exception as e:
        print(f"Error in enhanced allergen detection: {e}")
    
    # Supprimer les doublons en gardant le meilleur match
    unique_detected = []
    seen_allergens = set()
    
    for item in detected:
        key = f"{item['nom_fr']}_{item['nom_en']}_{item['nom_ar']}"
        if key not in seen_allergens:
            seen_allergens.add(key)
            unique_detected.append(item)
    
    return unique_detected

# Détection fuzzy (ancienne version pour compatibilité)
def detect_allergens(text, keywords, threshold=75):
    detected = []
    for keyword in keywords:
        for word in text.split():
            if fuzz.ratio(keyword, word) >= threshold:
                detected.append(keyword)
                break
    return list(set(detected))

# OCR image with optimizations
def extract_text_from_image(image):
    # Resize image if it's too large
    max_dimension = 1500
    width, height = image.size
    
    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        image = image.resize((new_width, new_height))
        print(f"Resized image to {new_width}x{new_height}")
    
    # Convert to grayscale for better OCR
    image = image.convert('L')
    
    # Improve contrast
    image = Image.fromarray(np.array(image))
    
    # Extract text with fallback languages
    start_time = time.time()
    
    # Try different language combinations
    language_options = [
        "eng+fra+ara",  # Preferred: English + French + Arabic
        "eng+fra",      # Fallback: English + French
        "eng",          # Fallback: English only
        ""              # Default language
    ]
    
    text = ""
    for lang in language_options:
        try:
            print(f"Trying OCR with language: {lang if lang else 'default'}")
            text = pytesseract.image_to_string(image, lang=lang if lang else None)
            print(f"OCR successful with language: {lang if lang else 'default'}")
            break
        except Exception as e:
            print(f"OCR failed with language '{lang}': {e}")
            continue
    
    if not text.strip():
        print("Warning: No text extracted from image")
        text = "No text detected"
    
    print(f"OCR processing took {time.time() - start_time:.2f} seconds")
    print('EXTRACTED RAW TEXT:\n', text)
    return text

# Process image with OCR
def process_image_ocr(image):
    # Extract text from image
    text = extract_text_from_image(image)
    print('EXTRACTED TEXT BEFORE CLEAN\n',text)
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Split into words
    words = cleaned_text.split()
    
    # Return the detected words
    return words if words else ["No text detected"]

# Endpoint pour récupérer tous les allergènes de la base de données
@app.route('/allergens', methods=['GET'])
def get_allergens():
    """Endpoint pour récupérer tous les allergènes de la base de données"""
    try:
        if df.empty:
            return jsonify({
                'status': 'error',
                'message': 'Allergens database not loaded'
            }), 500
        
        # Lecture du CSV et conversion en liste de dictionnaires
        allergens_list = []
        
        for index, row in df.iterrows():
            allergen_data = {
                'id': int(index + 1),
                'nom_fr': row.get('nom_fr', ''),
                'nom_en': row.get('nom_en', ''),
                'nom_ar': row.get('nom_ar', ''),
                'type': row.get('type', 'Allergène'),
                'sous_type_fr': row.get('sous_type_fr', ''),
                'sous_type_en': row.get('sous_type_en', ''),
                'sous_type_ar': row.get('sous_type_ar', ''),
                'code': row.get('code', ''),
                'proteine': row.get('proteine', '')
            }
            
            # Nettoyer les valeurs NaN
            for key, value in allergen_data.items():
                if pd.isna(value) or str(value).lower() == 'nan':
                    allergen_data[key] = ''
                else:
                    allergen_data[key] = str(value)
            
            allergens_list.append(allergen_data)
        
        return jsonify({
            'status': 'success',
            'allergens': allergens_list,
            'count': len(allergens_list)
        })
        
    except Exception as e:
        print(f"Error in get_allergens: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving allergens: {str(e)}'
        }), 500

# Endpoint pour rechercher des allergènes par nom
@app.route('/allergens/search', methods=['GET'])
def search_allergens():
    """Endpoint pour rechercher des allergènes par nom"""
    try:
        query = request.args.get('q', '').lower()
        
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query parameter "q" is required'
            }), 400
        
        if df.empty:
            return jsonify({
                'status': 'error',
                'message': 'Allergens database not loaded'
            }), 500
        
        # Filtrer les allergènes qui contiennent la requête
        matching_allergens = []
        
        for index, row in df.iterrows():
            # Vérifier dans toutes les colonnes de noms
            if (query in str(row.get('nom_fr', '')).lower() or
                query in str(row.get('nom_en', '')).lower() or
                query in str(row.get('nom_ar', '')) or
                query in str(row.get('code', '')).lower()):
                
                allergen_data = {
                    'id': int(index + 1),
                    'nom_fr': str(row.get('nom_fr', '')),
                    'nom_en': str(row.get('nom_en', '')),
                    'nom_ar': str(row.get('nom_ar', '')),
                    'type': str(row.get('type', 'Allergène')),
                    'sous_type_fr': str(row.get('sous_type_fr', '')),
                    'sous_type_en': str(row.get('sous_type_en', '')),
                    'sous_type_ar': str(row.get('sous_type_ar', '')),
                    'code': str(row.get('code', '')),
                    'proteine': str(row.get('proteine', ''))
                }
                
                # Nettoyer les valeurs NaN
                for key, value in allergen_data.items():
                    if pd.isna(value) or str(value).lower() == 'nan':
                        allergen_data[key] = ''
                
                matching_allergens.append(allergen_data)
        
        return jsonify({
            'status': 'success',
            'allergens': matching_allergens,
            'count': len(matching_allergens),
            'query': query
        })
        
    except Exception as e:
        print(f"Error in search_allergens: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error searching allergens: {str(e)}'
        }), 500

# Add a test endpoint
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly',
        'allergens_loaded': not df.empty,
        'allergens_count': len(df) if not df.empty else 0
    })

# Enhanced predict endpoint that handles both JSON and multipart requests
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        print("\n--- New request received ---")
        print(f"Content-Type: {request.content_type}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Request method: {request.method}")
        print(f"Request URL: {request.url}")
        
        start_time = time.time()
        image = None
        
        # Handle multipart form data (from Flutter)
        if request.content_type and 'multipart/form-data' in request.content_type:
            print("Processing multipart form data request (Flutter format)")
            
            if 'image' not in request.files:
                print("Error: No image file in multipart request")
                return jsonify({
                    'status': 'error', 
                    'message': 'No image file provided in multipart request'
                }), 400
            
            file = request.files['image']
            if file.filename == '':
                print("Error: Empty filename in multipart request")
                return jsonify({
                    'status': 'error', 
                    'message': 'No file selected'
                }), 400
            
            try:
                # Read the uploaded file directly into PIL Image
                print(f"Processing uploaded file: {file.filename}")
                image_data = file.read()
                print(f"File size: {len(image_data)} bytes")
                
                image = Image.open(io.BytesIO(image_data))
                print(f"Successfully opened image: {image.width}x{image.height}, mode: {image.mode}")
                
            except Exception as file_error:
                print(f"Error processing uploaded file: {file_error}")
                return jsonify({
                    'status': 'error', 
                    'message': f'Error processing uploaded file: {str(file_error)}'
                }), 400
        
        # Handle JSON request with base64 data
        elif request.is_json:
            print("Processing JSON request with base64 data")
            
            try:
                data = request.get_json()
                if data is None:
                    print("Error: request.get_json() returned None")
                    return jsonify({'status': 'error', 'message': 'Invalid JSON data'}), 400
                
                if 'image' not in data:
                    print("Error: No image in JSON request")
                    return jsonify({'status': 'error', 'message': 'No image provided in JSON'}), 400
                
                # Get the base64 image string
                image_data = data['image']
                print(f"Received base64 image (length: {len(image_data)})")
                
                # Handle data URL format (data:image/jpeg;base64,...)
                if image_data.startswith('data:'):
                    print("Detected data URL format, extracting base64 part...")
                    if ',' in image_data:
                        image_data = image_data.split(',', 1)[1]
                
                # Decode base64 image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                print(f"Successfully opened base64 image: {image.width}x{image.height}")
                
            except Exception as json_error:
                print(f"Error processing JSON request: {json_error}")
                return jsonify({
                    'status': 'error', 
                    'message': f'Error processing JSON request: {str(json_error)}'
                }), 400
        
        else:
            print(f"Error: Unsupported content type: {request.content_type}")
            return jsonify({
                'status': 'error',
                'message': 'Unsupported content type. Use multipart/form-data or application/json',
                'received_content_type': str(request.content_type)
            }), 400
        
        # Process the image with OCR
        if image is None:
            return jsonify({
                'status': 'error', 
                'message': 'Failed to load image'
            }), 400
        
        print(f"Processing image: {image.width}x{image.height}")
        
        # Resize image if it's too large
        if image.width > 1000 or image.height > 1000:
            resize_start = time.time()
            ratio = min(1000 / image.width, 1000 / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size)
            print(f"Resized image to {image.width}x{image.height} in {time.time() - resize_start:.2f} seconds")
        
        # Extract text using OCR
        ocr_start = time.time()
        extracted_text = extract_text_from_image(image)
        print(f"Extracted text in {time.time() - ocr_start:.2f} seconds")
        
        # Clean the text
        clean_start = time.time()
        cleaned_text = clean_text(extracted_text)
        detected_words = cleaned_text.split()
        print(f"Cleaned text in {time.time() - clean_start:.2f} seconds")
        
        # Detect allergens using enhanced method
        allergen_start = time.time()
        detected_allergens = detect_allergens_enhanced(cleaned_text)
        print(f"Enhanced allergen detection took {time.time() - allergen_start:.2f} seconds")
        print(f"Detected allergens: {[item['term'] for item in detected_allergens]}")
        
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
        
        # Return results in Flutter-compatible format
        return jsonify({
            'status': 'success',
            'ingredients': extracted_text if extracted_text.strip() else "No text detected",
            'detected_codes': detected_words if detected_words else ["No text detected"],
            'allergens': [item['term'] for item in detected_allergens],  # Pour compatibilité avec l'ancien code
            'allergen_details': detected_allergens,  # Détails complets des allergènes
            'hasAllergens': len(detected_allergens) > 0,
            'processing_time': round(time.time() - start_time, 2)
        })
        
    except Exception as e:
        print(f"Server error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error', 
            'message': f'Server error: {str(e)}'
        }), 500

# Keep the /ocr endpoint for compatibility
@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    print("OCR endpoint called, redirecting to predict endpoint...")
    return predict_endpoint()

# Add a debug endpoint to test connectivity
@app.route('/debug', methods=['GET', 'POST'])
def debug_endpoint():
    debug_info = {
        'status': 'success',
        'method': request.method,
        'headers': dict(request.headers),
        'content_type': request.content_type,
        'is_json': request.is_json,
        'data_length': len(request.data),
        'message': 'Debug endpoint working',
        'allergens_loaded': not df.empty,
        'allergens_count': len(df) if not df.empty else 0
    }
    
    if request.method == 'POST':
        if request.files:
            debug_info['files'] = list(request.files.keys())
        if request.form:
            debug_info['form_data'] = dict(request.form)
    
    return jsonify(debug_info)

if __name__ == '__main__':
    print("Starting OCR API server...")
    print("Available endpoints:")
    print("  GET  /test              - Test endpoint")
    print("  GET  /debug             - Debug endpoint")
    print("  GET  /allergens         - Get all allergens from database")
    print("  GET  /allergens/search  - Search allergens (use ?q=query)")
    print("  POST /predict           - Main OCR endpoint (supports both multipart and JSON)")
    print("  POST /ocr               - OCR endpoint (alias)")
    print("\nSupported request formats:")
    print("  1. Multipart form data with 'image' file (Flutter format)")
    print("  2. JSON with base64 'image' field")
    print(f"\nAllergens database status: {'Loaded' if not df.empty else 'Not loaded'}")
    if not df.empty:
        print(f"Total allergens in database: {len(df)}")
    
    # Use threaded=True for better handling of multiple requests
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)