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
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# Configuration Tesseract
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
CORS(app)

# Création du dossier temporaire s'il n'existe pas
if not os.path.exists("temp"):
    os.makedirs("temp")

def preprocess_image_for_ocr(image):
    """
    Préprocessing avancé de l'image pour améliorer l'OCR
    """
    print("Starting image preprocessing...")
    
    # Convertir PIL vers OpenCV
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Redimensionner si nécessaire (garder une bonne résolution)
    height, width = opencv_image.shape[:2]
    if width > 2000 or height > 2000:
        scale = min(2000/width, 2000/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        opencv_image = cv2.resize(opencv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"Resized to {new_width}x{new_height}")
    
    # 2. Convertir en niveaux de gris
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # 3. Réduire le bruit
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # 4. Améliorer le contraste avec CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # 5. Ajuster la luminosité si nécessaire
    mean_brightness = np.mean(contrast_enhanced)
    if mean_brightness < 100:  # Image trop sombre
        contrast_enhanced = cv2.convertScaleAbs(contrast_enhanced, alpha=1.2, beta=30)
        print("Image was too dark, brightness adjusted")
    elif mean_brightness > 200:  # Image trop claire
        contrast_enhanced = cv2.convertScaleAbs(contrast_enhanced, alpha=0.8, beta=-20)
        print("Image was too bright, brightness adjusted")
    
    # 6. Appliquer un filtre de netteté
    kernel_sharpening = np.array([[-1,-1,-1],
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel_sharpening)
    
    # 7. Binarisation adaptative pour améliorer la lisibilité du texte
    # Essayer différentes méthodes de binarisation
    binary_adaptive = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 8. Opérations morphologiques pour nettoyer le texte
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel)
    
    # Convertir de retour en PIL Image
    processed_image = Image.fromarray(cleaned)
    
    print("Image preprocessing completed")
    return processed_image, opencv_image

def extract_text_from_image_enhanced(image):
    """
    Extraction de texte améliorée avec préprocessing
    """
    start_time = time.time()
    
    # Préprocesser l'image
    processed_image, original_cv = preprocess_image_for_ocr(image)
    
    # Essayer différentes configurations OCR
    ocr_configs = [
        # Configuration pour texte multilingue avec amélioration
        {
            'lang': 'eng+fra+ara',
            'config': '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789àâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ .,()-/'
        },
        # Configuration pour texte en blocs
        {
            'lang': 'eng+fra+ara',
            'config': '--psm 3 --oem 3'
        },
        # Configuration pour lignes de texte
        {
            'lang': 'eng+fra+ara',
            'config': '--psm 7 --oem 3'
        },
        # Configuration de base
        {
            'lang': 'eng+fra+ara',
            'config': '--psm 6 --oem 3'
        }
    ]
    
    best_text = ""
    best_confidence = 0
    
    for config in ocr_configs:
        try:
            print(f"Trying OCR with config: {config['config']}")
            
            # Extraire le texte avec la configuration actuelle
            text = pytesseract.image_to_string(
                processed_image, 
                lang=config['lang'], 
                config=config['config']
            )
            
            # Calculer une estimation de confiance basée sur la longueur et la qualité du texte
            confidence = len(text.strip()) * (len([c for c in text if c.isalnum()]) / max(len(text), 1))
            
            print(f"Extracted text length: {len(text)}, confidence: {confidence:.2f}")
            
            if confidence > best_confidence and len(text.strip()) > 0:
                best_text = text
                best_confidence = confidence
                print(f"New best result with confidence: {confidence:.2f}")
            
        except Exception as e:
            print(f"OCR failed with config {config}: {e}")
            continue
    
    # Si aucun résultat satisfaisant, essayer avec l'image originale
    if best_confidence < 10:
        print("Low confidence, trying with original image...")
        try:
            best_text = pytesseract.image_to_string(image, lang='eng+fra+ara')
        except Exception as e:
            print(f"OCR with original image failed: {e}")
    
    if not best_text.strip():
        print("Warning: No text extracted from image")
        best_text = "No text detected"
    
    print(f"Total OCR processing took {time.time() - start_time:.2f} seconds")
    print(f'EXTRACTED RAW TEXT (confidence: {best_confidence:.2f}):\n{best_text}')
    
    return best_text

def clean_text_enhanced(text):
    """
    Nettoyage de texte amélioré
    """
    # Nettoyer les caractères de contrôle et les espaces multiples
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    
    # Nettoyer les caractères spéciaux mais garder les accents et caractères arabes
    text = re.sub(r"[^\w\sÀ-ÿ\u0600-\u06FF\u0750-\u077F.,()-/]", " ", text)
    
    # Remplacer les espaces multiples par un seul espace
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les espaces en début et fin
    text = text.strip()
    
    return text.lower()

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
    df = pd.DataFrame()

def detect_allergens_enhanced(text, threshold=75):
    """
    Détection d'allergènes avec seuil de confiance ajusté
    """
    detected = []
    text_lower = text.lower()
    
    try:
        for index, row in df.iterrows():
            terms_to_check = []
            
            # Ajouter les noms dans différentes langues
            for col in ['nom_fr', 'nom_en', 'nom_ar', 'sous_type_fr', 'sous_type_en', 'sous_type_ar', 'code']:
                if col in df.columns and pd.notna(row[col]):
                    term = str(row[col]).strip()
                    if term and term.lower() != 'nan':
                        terms_to_check.append(term)
            
            # Vérifier chaque terme
            for term in terms_to_check:
                term_lower = term.lower()
                
                # Recherche exacte d'abord
                if term_lower in text_lower:
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
                        'match_type': 'exact',
                        'confidence': 100
                    })
                    break
                
                # Recherche fuzzy avec seuil adaptatif
                else:
                    # Utiliser un seuil plus bas pour les mots courts
                    adaptive_threshold = max(threshold - (10 if len(term_lower) < 5 else 0), 60)
                    
                    for word in text_lower.split():
                        # Nettoyer le mot pour la comparaison
                        clean_word = re.sub(r'[^\w]', '', word)
                        if len(clean_word) > 1:
                            similarity = fuzz.ratio(term_lower, clean_word)
                            if similarity >= adaptive_threshold:
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
                                    'confidence': similarity
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
        elif item['confidence'] > next((x['confidence'] for x in unique_detected if f"{x['nom_fr']}_{x['nom_en']}_{x['nom_ar']}" == key), 0):
            # Remplacer par un match de meilleure qualité
            unique_detected = [x for x in unique_detected if f"{x['nom_fr']}_{x['nom_en']}_{x['nom_ar']}" != key]
            unique_detected.append(item)
    
    return unique_detected

# Endpoints restants (complets)
@app.route('/allergens', methods=['GET'])
def get_allergens():
    """Endpoint pour récupérer tous les allergènes de la base de données"""
    try:
        if df.empty:
            return jsonify({
                'status': 'error',
                'message': 'Allergens database not loaded'
            }), 500
        
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
        
        matching_allergens = []
        
        for index, row in df.iterrows():
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

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'status': 'success',
        'message': 'Enhanced OCR API is working correctly',
        'allergens_loaded': not df.empty,
        'allergens_count': len(df) if not df.empty else 0,
        'opencv_available': True
    })

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        print("\n--- New request received ---")
        print(f"Content-Type: {request.content_type}")
        
        start_time = time.time()
        image = None
        
        # Handle multipart form data (from Flutter)
        if request.content_type and 'multipart/form-data' in request.content_type:
            print("Processing multipart form data request (Flutter format)")
            
            if 'image' not in request.files:
                return jsonify({
                    'status': 'error', 
                    'message': 'No image file provided in multipart request'
                }), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'status': 'error', 
                    'message': 'No file selected'
                }), 400
            
            try:
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
                if data is None or 'image' not in data:
                    return jsonify({'status': 'error', 'message': 'Invalid JSON data or no image provided'}), 400
                
                image_data = data['image']
                
                # Handle data URL format
                if image_data.startswith('data:'):
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
            return jsonify({
                'status': 'error',
                'message': 'Unsupported content type. Use multipart/form-data or application/json'
            }), 400
        
        # Process the image with enhanced OCR
        if image is None:
            return jsonify({
                'status': 'error', 
                'message': 'Failed to load image'
            }), 400
        
        print(f"Processing image: {image.width}x{image.height}")
        
        # Extract text using enhanced OCR
        extracted_text = extract_text_from_image_enhanced(image)
        
        # Clean the text with enhanced cleaning
        cleaned_text = clean_text_enhanced(extracted_text)
        detected_words = cleaned_text.split()
        
        # Detect allergens using enhanced method
        detected_allergens = detect_allergens_enhanced(cleaned_text, threshold=70)  # Seuil légèrement plus bas
        
        print(f"Detected allergens: {[item['term'] for item in detected_allergens]}")
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
        
        # Return results
        return jsonify({
            'status': 'success',
            'ingredients': extracted_text if extracted_text.strip() else "No text detected",
            'detected_codes': detected_words if detected_words else ["No text detected"],
            'allergens': [item['term'] for item in detected_allergens],
            'allergen_details': detected_allergens,
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

# Endpoint pour l'ancien système (compatibilité)
@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    print("OCR endpoint called, redirecting to predict endpoint...")
    return predict_endpoint()

# Endpoint de debug
@app.route('/debug', methods=['GET', 'POST'])
def debug_endpoint():
    debug_info = {
        'status': 'success',
        'method': request.method,
        'headers': dict(request.headers),
        'content_type': request.content_type,
        'is_json': request.is_json,
        'data_length': len(request.data),
        'message': 'Enhanced Debug endpoint working',
        'allergens_loaded': not df.empty,
        'allergens_count': len(df) if not df.empty else 0,
        'opencv_available': True
    }
    
    if request.method == 'POST':
        if request.files:
            debug_info['files'] = list(request.files.keys())
        if request.form:
            debug_info['form_data'] = dict(request.form)
    
    return jsonify(debug_info)
    print("Starting Enhanced OCR API server...")
    print("Enhanced features:")
    print("  - Advanced image preprocessing")
    print("  - Noise reduction and contrast enhancement")
    print("  - Multiple OCR configurations")
    print("  - Adaptive thresholding for better text detection")
    print("  - Improved fuzzy matching with adaptive thresholds")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)