from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import arabic_reshaper
from bidi.algorithm import get_display
import os
import cv2
import numpy as np
import re

app = Flask(__name__)

# Configuration Tesseract pour Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly'
    })

@app.route('/predict', methods=['POST'])
def extract_ingredients():
    try:
        image = request.files['image']
        print(f"Received image: {image.filename}")
        
        # Sauvegarder l'image temporairement
        temp_path = os.path.join(os.getcwd(), image.filename)
        image.save(temp_path)
        
        # Créer le dossier pour les images prétraitées
        preprocess_dir = os.path.join(os.getcwd(), "preprocessed")
        if not os.path.exists(preprocess_dir):
            os.makedirs(preprocess_dir)
        
        # Lire l'image
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({'error': 'Failed to read the image'}), 500
        
        # Vérifier les langues disponibles
        try:
            available_langs = pytesseract.get_languages()
            print(f"Available languages: {available_langs}")
        except:
            available_langs = ['eng']  # Fallback à l'anglais seulement
        
        # Prétraitement avancé de l'image
        processed_images = advanced_preprocessing(img, image.filename, preprocess_dir)
        
        # Extraire le texte avec différentes configurations
        all_texts = []
        
        for img_path in processed_images:
            # Configuration 1: PSM 6 (bloc de texte uniforme)
            try:
                text1 = pytesseract.image_to_string(
                    Image.open(img_path),
                    lang='eng',
                    config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.:-% '
                )
                all_texts.append(text1)
            except Exception as e:
                print(f"Error with config 1: {e}")
            
            # Configuration 2: PSM 4 (colonne de texte)
            try:
                text2 = pytesseract.image_to_string(
                    Image.open(img_path),
                    lang='eng',
                    config='--psm 4'
                )
                all_texts.append(text2)
            except Exception as e:
                print(f"Error with config 2: {e}")
            
            # Configuration 3: Spécifique à l'arabe si disponible
            if 'ara' in available_langs:
                try:
                    text3 = pytesseract.image_to_string(
                        Image.open(img_path),
                        lang='ara',
                        config='--psm 6'
                    )
                    all_texts.append(text3)
                except Exception as e:
                    print(f"Error with Arabic config: {e}")
                    
            
            # Configuration 4: Multilingue si disponible
            try:
                if 'ara' in available_langs and 'fra' in available_langs:
                    lang_config = 'eng+ara+fra'
                elif 'ara' in available_langs:
                    lang_config = 'eng+ara'
                elif 'fra' in available_langs:
                    lang_config = 'eng+fra'
                else:
                    lang_config = 'eng'
                
                text4 = pytesseract.image_to_string(
                    Image.open(img_path),
                    lang=lang_config,
                    config='--psm 6'
                )
                all_texts.append(text4)
            except Exception as e:
                print(f"Error with multilingual config: {e}")
        
        # Combiner et nettoyer tous les textes
        combined_text = "\n".join(filter(None, all_texts))
        print(f"Raw OCR text: {combined_text}")
        
        # Extraire la section des ingrédients
        ingredients_text = extract_ingredients_section_advanced(combined_text)
        
        # Traitement du texte arabe
        if ingredients_text and any(is_arabic_character(c) for c in ingredients_text):
            try:
                ingredients_text = get_display(arabic_reshaper.reshape(ingredients_text))
            except Exception as e:
                print(f"Arabic processing error: {e}")
        
        # Nettoyage final
        ingredients_text = clean_text(ingredients_text)
        
        # Nettoyage des fichiers temporaires
        cleanup_files(temp_path, processed_images)
        
        return jsonify({'ingredients': ingredients_text})
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def advanced_preprocessing(img, filename, output_dir):
    """Prétraitement avancé pour améliorer la qualité OCR"""
    output_paths = []
    base_name = os.path.splitext(filename)[0]
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Image originale en niveaux de gris
    path1 = os.path.join(output_dir, f"{base_name}_gray.png")
    cv2.imwrite(path1, gray)
    output_paths.append(path1)
    
    # 2. Amélioration du contraste avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    path2 = os.path.join(output_dir, f"{base_name}_clahe.png")
    cv2.imwrite(path2, cl1)
    output_paths.append(path2)
    
    # 3. Débruitage
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    path3 = os.path.join(output_dir, f"{base_name}_denoised.png")
    cv2.imwrite(path3, denoised)
    output_paths.append(path3)
    
    # 4. Seuillage OTSU
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    path4 = os.path.join(output_dir, f"{base_name}_otsu.png")
    cv2.imwrite(path4, thresh_otsu)
    output_paths.append(path4)
    
    # 5. Seuillage adaptatif
    thresh_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    path5 = os.path.join(output_dir, f"{base_name}_adaptive.png")
    cv2.imwrite(path5, thresh_adaptive)
    output_paths.append(path5)
    
    # 6. Agrandissement de l'image
    height, width = gray.shape
    resized = cv2.resize(gray, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
    clahe_resized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    resized_enhanced = clahe_resized.apply(resized)
    path6 = os.path.join(output_dir, f"{base_name}_resized.png")
    cv2.imwrite(path6, resized_enhanced)
    output_paths.append(path6)
    
    # 7. Amélioration avec PIL
    try:
        pil_img = Image.fromarray(gray)
        
        # Amélioration de la netteté
        enhancer = ImageEnhance.Sharpness(pil_img)
        sharp_img = enhancer.enhance(2.0)
        
        # Amélioration du contraste
        contrast_enhancer = ImageEnhance.Contrast(sharp_img)
        contrast_img = contrast_enhancer.enhance(1.5)
        
        path7 = os.path.join(output_dir, f"{base_name}_pil_enhanced.png")
        contrast_img.save(path7)
        output_paths.append(path7)
    except Exception as e:
        print(f"PIL enhancement error: {e}")
    
    return output_paths

def extract_ingredients_section_advanced(text):
    """Extraction avancée de la section des ingrédients"""
    if not text or len(text.strip()) < 3:
        return "Aucun texte détecté"
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Marqueurs pour identifier la section des ingrédients
    ingredient_markers = [
        r'ingredients?:?',
        r'ingrédients?:?',
        r'المكونات:?',
        r'مكونات:?',
        r'composition:?',
        r'ingredientes?:?'
    ]
    
    # Marqueurs de fin de section
    end_markers = [
        r'nutrition',
        r'valeur nutritive',
        r'القيمة الغذائية',
        r'allergène',
        r'allergen',
        r'conservation',
        r'storage',
        r'best before',
        r'expiry'
    ]
    
    # Recherche de la section des ingrédients
    start_idx = -1
    end_idx = len(lines)
    
    # Trouver le début
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for marker in ingredient_markers:
            if re.search(marker, line_lower):
                start_idx = i
                break
        if start_idx >= 0:
            break
    
    # Si on n'a pas trouvé de marqueur, chercher une ligne avec beaucoup d'ingrédients
    if start_idx < 0:
        common_ingredients = [
            'sucre', 'sugar', 'eau', 'water', 'farine', 'flour', 'huile', 'oil',
            'lait', 'milk', 'sel', 'salt', 'ماء', 'سكر', 'ملح', 'زيت'
        ]
        
        best_score = 0
        for i, line in enumerate(lines):
            score = sum(1 for ing in common_ingredients if ing.lower() in line.lower())
            if score > best_score and score >= 2:
                best_score = score
                start_idx = i
    
    # Trouver la fin
    if start_idx >= 0:
        for i in range(start_idx + 1, len(lines)):
            line_lower = lines[i].lower()
            for marker in end_markers:
                if re.search(marker, line_lower):
                    end_idx = i
                    break
            if end_idx < len(lines):
                break
    
    # Extraire la section
    if start_idx >= 0:
        ingredient_lines = lines[start_idx:end_idx]
        # Limiter à 10 lignes maximum pour éviter d'inclure trop de texte
        ingredient_lines = ingredient_lines[:10]
        return '\n'.join(ingredient_lines)
    
    # Si rien n'est trouvé, retourner les premières lignes
    return '\n'.join(lines[:5]) if lines else "Aucun ingrédient détecté"

def clean_text(text):
    """Nettoyage final du texte"""
    if not text:
        return "Aucun texte détecté"
    
    # Supprimer les caractères de contrôle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Supprimer les lignes vides multiples
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Supprimer les lignes trop courtes (probablement des erreurs OCR)
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
        print(f"Error cleaning up files: {e}")

if __name__ == '__main__':
    # Vérification de Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        languages = pytesseract.get_languages()
        print(f"Tesseract version: {version}")
        print(f"Available languages: {languages}")
    except Exception as e:
        print(f"Tesseract issue: {e}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)