from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import os
import re
from PIL import Image

app = Flask(__name__)

# Initialiser PaddleOCR une seule fois
print("Initialisation de PaddleOCR...")
try:
    # PaddleOCR avec support multilingue
    # lang='en' pour anglais, 'ch' pour chinois, 'french' pour français, 'arabic' pour arabe
    ocr_en = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)  # Anglais
    ocr_ar = None
    ocr_fr = None
    
    # Essayer d'initialiser l'arabe (peut ne pas être disponible)
    try:
        ocr_ar = PaddleOCR(use_angle_cls=True, lang='arabic', use_gpu=False)
        print("PaddleOCR arabe initialisé!")
    except:
        print("PaddleOCR arabe non disponible")
    
    # Essayer d'initialiser le français
    try:
        ocr_fr = PaddleOCR(use_angle_cls=True, lang='french', use_gpu=False)
        print("PaddleOCR français initialisé!")
    except:
        print("PaddleOCR français non disponible")
    
    print("PaddleOCR initialisé avec succès!")
except Exception as e:
    print(f"Erreur lors de l'initialisation de PaddleOCR: {e}")
    ocr_en = None

@app.route('/test', methods=['GET'])
def test_endpoint():
    status = 'success' if ocr_en is not None else 'error'
    message = 'API is working correctly with PaddleOCR' if ocr_en else 'PaddleOCR not initialized'
    return jsonify({
        'status': status,
        'message': message
    })

@app.route('/predict', methods=['POST'])
def extract_ingredients():
    try:
        if ocr_en is None:
            return jsonify({'error': 'PaddleOCR not initialized'}), 500

        # Récupérer l'image
        image = request.files['image']
        print(f"Received image: {image.filename}")
        
        # Sauvegarder temporairement
        temp_path = os.path.join(os.getcwd(), image.filename)
        image.save(temp_path)
        
        # Créer le dossier pour les images prétraitées
        preprocess_dir = os.path.join(os.getcwd(), "preprocessed")
        if not os.path.exists(preprocess_dir):
            os.makedirs(preprocess_dir)
        
        # Lire l'image avec OpenCV
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({'error': 'Failed to read the image'}), 500
        
        # Prétraitement de l'image
        processed_images = preprocess_for_paddleocr(img, image.filename, preprocess_dir)
        
        # Extraire le texte avec PaddleOCR
        all_texts = []
        
        for img_path in processed_images:
            try:
                print(f"Processing {img_path} with PaddleOCR...")
                
                # Essayer avec différents modèles de langue
                ocr_results = []
                
                # 1. Essayer avec l'anglais
                try:
                    result_en = ocr_en.ocr(img_path, cls=True)
                    if result_en and result_en[0]:
                        ocr_results.extend(result_en[0])
                        print("Texte détecté avec modèle anglais")
                except Exception as e:
                    print(f"Erreur avec modèle anglais: {e}")
                
                # 2. Essayer avec l'arabe si disponible
                if ocr_ar:
                    try:
                        result_ar = ocr_ar.ocr(img_path, cls=True)
                        if result_ar and result_ar[0]:
                            ocr_results.extend(result_ar[0])
                            print("Texte détecté avec modèle arabe")
                    except Exception as e:
                        print(f"Erreur avec modèle arabe: {e}")
                
                # 3. Essayer avec le français si disponible
                if ocr_fr:
                    try:
                        result_fr = ocr_fr.ocr(img_path, cls=True)
                        if result_fr and result_fr[0]:
                            ocr_results.extend(result_fr[0])
                            print("Texte détecté avec modèle français")
                    except Exception as e:
                        print(f"Erreur avec modèle français: {e}")
                
                # Extraire le texte des résultats
                extracted_texts = []
                for result in ocr_results:
                    if result and len(result) >= 2:
                        bbox, (text, confidence) = result
                        if confidence > 0.5:  # Seuil de confiance
                            extracted_texts.append(text)
                            print(f"Texte: '{text}' (confiance: {confidence:.2f})")
                
                # Joindre tous les textes de cette image
                if extracted_texts:
                    # Trier par position verticale (approximative)
                    combined_text = ' '.join(extracted_texts)
                    all_texts.append(combined_text)
                    
            except Exception as e:
                print(f"Erreur PaddleOCR avec {img_path}: {e}")
        
        # Combiner tous les résultats
        if all_texts:
            full_text = '\n'.join(all_texts)
            print(f"Texte complet détecté:\n{full_text}")
        else:
            full_text = ""
        
        # Extraire la section des ingrédients
        ingredients_text = extract_ingredients_section_advanced(full_text)
        
        # Traitement du texte arabe si nécessaire
        if ingredients_text and any(is_arabic_character(c) for c in ingredients_text):
            try:
                ingredients_text = get_display(arabic_reshaper.reshape(ingredients_text))
            except Exception as e:
                print(f"Erreur traitement arabe: {e}")
        
        # Nettoyage final
        ingredients_text = clean_and_format_text(ingredients_text)
        
        # Nettoyage des fichiers temporaires
        cleanup_files(temp_path, processed_images)
        
        return jsonify({'ingredients': ingredients_text})
        
    except Exception as e:
        print(f"Erreur générale: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def preprocess_for_paddleocr(img, filename, output_dir):
    """Prétraitement optimisé pour PaddleOCR"""
    output_paths = []
    base_name = os.path.splitext(filename)[0]
    
    # 1. Image originale
    original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
    cv2.imwrite(original_path, img)
    output_paths.append(original_path)
    
    # 2. Amélioration de la luminosité et du contraste
    alpha = 1.2  # Contraste
    beta = 30    # Luminosité
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    output_paths.append(enhanced_path)
    
    # 3. Égalisation d'histogramme adaptatif
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    clahe_img = cv2.merge([l, a, b])
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)
    
    clahe_path = os.path.join(output_dir, f"{base_name}_clahe.jpg")
    cv2.imwrite(clahe_path, clahe_img)
    output_paths.append(clahe_path)
    
    # 4. Débruitage
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    denoised_path = os.path.join(output_dir, f"{base_name}_denoised.jpg")
    cv2.imwrite(denoised_path, denoised)
    output_paths.append(denoised_path)
    
    # 5. Netteté améliorée
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    sharp_path = os.path.join(output_dir, f"{base_name}_sharp.jpg")
    cv2.imwrite(sharp_path, sharpened)
    output_paths.append(sharp_path)
    
    return output_paths

def extract_ingredients_section_advanced(text):
    """Extraction intelligente de la section des ingrédients"""
    if not text or len(text.strip()) < 3:
        return "Aucun texte détecté"
    
    # Nettoyer le texte d'abord
    text = re.sub(r'\s+', ' ', text)  # Normaliser les espaces
    
    # Marqueurs d'ingrédients multilingues
    ingredient_patterns = [
        r'ingredients?\s*:',
        r'ingrédients?\s*:',
        r'المكونات\s*:?',
        r'مكونات\s*:?',
        r'composition\s*:',
        r'ingredientes?\s*:',
        r'قائمة المكونات'
    ]
    
    best_match = ""
    max_length = 0
    
    # Chercher chaque pattern
    for pattern in ingredient_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            start_pos = match.start()
            # Prendre du texte après le marqueur (jusqu'à 500 caractères)
            end_pos = min(start_pos + 500, len(text))
            candidate = text[start_pos:end_pos]
            
            # Arrêter si on trouve un marqueur de fin
            end_patterns = [r'nutrition', r'valeur nutritive', r'القيمة الغذائية', 
                          r'allergène', r'allergen', r'conservation']
            for end_pattern in end_patterns:
                end_match = re.search(end_pattern, candidate, re.IGNORECASE)
                if end_match:
                    candidate = candidate[:end_match.start()]
                    break
            
            if len(candidate) > max_length:
                best_match = candidate
                max_length = len(candidate)
    
    # Si on n'a pas trouvé de marqueur, chercher par mots-clés d'ingrédients
    if not best_match:
        ingredient_keywords = [
            'sucre', 'sugar', 'eau', 'water', 'farine', 'flour', 
            'huile', 'oil', 'lait', 'milk', 'sel', 'salt',
            'ماء', 'سكر', 'ملح', 'زيت', 'حليب'
        ]
        
        # Diviser le texte en segments
        segments = text.split('.')
        best_score = 0
        
        for segment in segments:
            score = sum(1 for keyword in ingredient_keywords if keyword.lower() in segment.lower())
            if score > best_score:
                best_score = score
                best_match = segment
    
    return best_match.strip() if best_match else "Aucun ingrédient détecté"

def clean_and_format_text(text):
    """Nettoyage et formatage final du texte"""
    if not text:
        return "Aucun texte détecté"
    
    # Supprimer les caractères de contrôle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les caractères répétés bizarres
    text = re.sub(r'(.)\1{4,}', r'\1', text)
    
    return text.strip() if text.strip() else "Texte non valide détecté"

def is_arabic_character(char):
    """Vérifier si un caractère est arabe"""
    return '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or '\u08A0' <= char <= '\u08FF'

def cleanup_files(temp_path, processed_paths):
    """Nettoyage des fichiers temporaires"""
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        for path in processed_paths:
            if os.path.exists(path):
                os.remove(path)
    except Exception as e:
        print(f"Erreur nettoyage fichiers: {e}")

if __name__ == '__main__':
    print("Démarrage du serveur Flask avec PaddleOCR...")
    app.run(host='0.0.0.0', port=5000, debug=True)