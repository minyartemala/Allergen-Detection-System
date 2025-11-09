from flask import Flask, request, jsonify
import easyocr
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import os
import re
from PIL import Image, ImageEnhance

app = Flask(__name__)

# Initialiser EasyOCR une seule fois (pour éviter de le recharger à chaque requête)
# Langues supportées : 'en' (anglais), 'ar' (arabe), 'fr' (français)
print("Initialisation d'EasyOCR...")
try:
    # Créer le lecteur EasyOCR avec support multilingue
    reader = easyocr.Reader(['en', 'ar', 'fr'], gpu=False)  # gpu=True si vous avez une GPU
    print("EasyOCR initialisé avec succès!")
except Exception as e:
    print(f"Erreur lors de l'initialisation d'EasyOCR: {e}")
    reader = None

@app.route('/test', methods=['GET'])
def test_endpoint():
    status = 'success' if reader is not None else 'error'
    message = 'API is working correctly with EasyOCR' if reader else 'EasyOCR not initialized'
    return jsonify({
        'status': status,
        'message': message
    })

@app.route('/predict', methods=['POST'])
def extract_ingredients():
    try:
        if reader is None:
            return jsonify({'error': 'EasyOCR not initialized'}), 500

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
        processed_images = preprocess_for_easyocr(img, image.filename, preprocess_dir)
        
        # Extraire le texte avec EasyOCR
        all_results = []
        
        for img_path in processed_images:
            try:
                print(f"Processing {img_path} with EasyOCR...")
                
                # EasyOCR retourne une liste de (bbox, texte, confiance)
                results = reader.readtext(img_path)
                
                # Extraire seulement le texte avec une confiance suffisante
                extracted_texts = []
                for (bbox, text, confidence) in results:
                    if confidence > 0.3:  # Seuil de confiance (ajustable)
                        extracted_texts.append(text)
                        print(f"Texte détecté: '{text}' (confiance: {confidence:.2f})")
                
                # Joindre tous les textes de cette image
                if extracted_texts:
                    combined_text = ' '.join(extracted_texts)
                    all_results.append(combined_text)
                    
            except Exception as e:
                print(f"Erreur EasyOCR avec {img_path}: {e}")
        
        # Combiner tous les résultats
        if all_results:
            full_text = '\n'.join(all_results)
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

def preprocess_for_easyocr(img, filename, output_dir):
    """Prétraitement optimisé pour EasyOCR"""
    output_paths = []
    base_name = os.path.splitext(filename)[0]
    
    # 1. Image originale
    original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
    cv2.imwrite(original_path, img)
    output_paths.append(original_path)
    
    # 2. Amélioration du contraste
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    output_paths.append(enhanced_path)
    
    # 3. Redimensionnement (EasyOCR fonctionne mieux avec des images plus grandes)
    height, width = img.shape[:2]
    if height < 800 or width < 800:  # Agrandir si l'image est petite
        scale_factor = max(800/height, 800/width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        resized_path = os.path.join(output_dir, f"{base_name}_resized.jpg")
        cv2.imwrite(resized_path, resized)
        output_paths.append(resized_path)
    
    # 4. Débruitage
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    denoised_path = os.path.join(output_dir, f"{base_name}_denoised.jpg")
    cv2.imwrite(denoised_path, denoised)
    output_paths.append(denoised_path)
    
    # 5. Version en niveaux de gris avec amélioration
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.equalizeHist(gray)  # Égalisation d'histogramme
    gray_bgr = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)  # EasyOCR préfère les images couleur
    
    gray_path = os.path.join(output_dir, f"{base_name}_gray_enhanced.jpg")
    cv2.imwrite(gray_path, gray_bgr)
    output_paths.append(gray_path)
    
    return output_paths

def extract_ingredients_section_advanced(text):
    """Extraction intelligente de la section des ingrédients"""
    if not text or len(text.strip()) < 3:
        return "Aucun texte détecté"
    
    # Nettoyer le texte d'abord
    text = re.sub(r'\s+', ' ', text)  # Normaliser les espaces
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Marqueurs d'ingrédients en plusieurs langues
    ingredient_markers = [
        r'ingredients?:?',
        r'ingrédients?:?',
        r'المكونات:?',
        r'مكونات:?',
        r'composition:?',
        r'ingredientes?:?',
        r'قائمة المكونات'
    ]
    
    # Marqueurs de fin
    end_markers = [
        r'nutrition', r'valeur nutritive', r'القيمة الغذائية',
        r'allergène', r'allergen', r'حساسية',
        r'conservation', r'storage', r'حفظ',
        r'best before', r'expiry', r'تاريخ الانتهاء'
    ]
    
    # Rechercher la section des ingrédients
    best_section = ""
    max_score = 0
    
    # Méthode 1: Chercher par marqueurs
    for line in lines:
        line_lower = line.lower()
        for marker in ingredient_markers:
            if re.search(marker, line_lower):
                # Prendre cette ligne et quelques suivantes
                idx = lines.index(line)
                section = []
                section.append(line)
                
                # Ajouter les lignes suivantes jusqu'à un marqueur de fin
                for i in range(idx + 1, min(idx + 8, len(lines))):
                    next_line = lines[i].lower()
                    if any(re.search(end_marker, next_line) for end_marker in end_markers):
                        break
                    section.append(lines[i])
                
                candidate = '\n'.join(section)
                if len(candidate) > len(best_section):
                    best_section = candidate
    
    # Méthode 2: Chercher par densité d'ingrédients courants
    if not best_section:
        common_ingredients = [
            'sucre', 'sugar', 'eau', 'water', 'farine', 'flour', 
            'huile', 'oil', 'lait', 'milk', 'sel', 'salt',
            'ماء', 'سكر', 'ملح', 'زيت', 'حليب', 'دقيق'
        ]
        
        for i, line in enumerate(lines):
            score = sum(1 for ing in common_ingredients if ing.lower() in line.lower())
            if score > max_score:
                max_score = score
                # Prendre cette ligne et quelques suivantes
                section_lines = lines[i:min(i+5, len(lines))]
                best_section = '\n'.join(section_lines)
    
    # Si toujours rien, prendre les premières lignes significatives
    if not best_section and lines:
        best_section = '\n'.join(lines[:min(3, len(lines))])
    
    return best_section if best_section else "Aucun ingrédient détecté"

def clean_and_format_text(text):
    """Nettoyage et formatage final du texte"""
    if not text:
        return "Aucun texte détecté"
    
    # Supprimer les caractères de contrôle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Séparer les lignes
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Supprimer les lignes trop courtes (probablement des erreurs)
    lines = [line for line in lines if len(line) > 2]
    
    return '\n'.join(lines) if lines else "Aucun ingrédient valide détecté"

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
    print("Démarrage du serveur Flask avec EasyOCR...")
    app.run(host='0.0.0.0', port=5000, debug=True)