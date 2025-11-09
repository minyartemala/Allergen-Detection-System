import base64
import requests
import os
from mistralai import Mistral

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def detect_allergens(image_path, allergens_list=None):
    """
    Détecte les allergènes dans une image de produit alimentaire
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        allergens_list (list): Liste d'allergènes spécifiques à rechercher
    
    Returns:
        dict: Résultats de la détection d'allergènes
    """
    
    # Liste des allergènes les plus courants si non spécifiée
    if allergens_list is None:
        allergens_list = [
            "gluten", "blé", "seigle", "orge", "avoine",
            "lait", "lactose", "caséine",
            "œufs", "oeufs",
            "poisson", "crustacés", "mollusques",
            "arachides", "cacahuètes",
            "fruits à coque", "noix", "amandes", "noisettes", "pistaches",
            "soja", "soya",
            "céleri",
            "moutarde",
            "graines de sésame", "sésame",
            "anhydride sulfureux", "sulfites",
            "lupin"
        ]
    
    # Encoder l'image
    base64_image = encode_image(image_path)
    if not base64_image:
        return {"error": "Impossible d'encoder l'image"}
    
    # Configuration de l'API
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return {"error": "Clé API Mistral non trouvée dans les variables d'environnement"}
    
    model = "mistral-small-latest"
    client = Mistral(api_key=api_key)
    
    # Créer le prompt spécialisé pour la détection d'allergènes
    allergens_str = ", ".join(allergens_list)
    
    prompt = f"""
    Analysez cette image de produit alimentaire et détectez la présence d'allergènes.
    
    Recherchez spécifiquement ces allergènes : {allergens_str}
    
    Instructions :
    1. Lisez attentivement tous les textes visibles sur l'emballage (étiquettes, listes d'ingrédients, mentions d'allergènes)
    2. Identifiez les allergènes présents dans la liste d'ingrédients
    3. Repérez les mentions "Peut contenir", "Traces de", "Fabriqué dans un atelier qui utilise"
    4. Fournissez une réponse structurée avec :
       - ALLERGÈNES DÉTECTÉS : [liste des allergènes trouvés]
       - TRACES POSSIBLES : [allergènes mentionnés comme traces]
       - NIVEAU DE RISQUE : [Élevé/Moyen/Faible]
       - RECOMMANDATIONS : [conseils pour les personnes allergiques]
    
    Si vous ne pouvez pas lire clairement le texte, indiquez-le.
    """
    
    # Messages pour l'API
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    
    try:
        # Appeler l'API
        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )
        
        response_text = chat_response.choices[0].message.content
        
        # Parser la réponse (optionnel : vous pouvez améliorer cette partie)
        result = {
            "status": "success",
            "analysis": response_text,
            "image_path": image_path
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Erreur lors de l'appel à l'API : {str(e)}"}

def batch_detect_allergens(image_paths, allergens_list=None):
    """
    Détecte les allergènes dans plusieurs images
    
    Args:
        image_paths (list): Liste des chemins d'images
        allergens_list (list): Liste d'allergènes à rechercher
    
    Returns:
        list: Résultats pour chaque image
    """
    results = []
    for image_path in image_paths:
        print(f"Analyse de {image_path}...")
        result = detect_allergens(image_path, allergens_list)
        results.append(result)
    return results

# Exemple d'utilisation
if __name__ == "__main__":
    # Chemin vers votre image
    image_path = "baristella rocher.jpeg"
    
    # Allergènes spécifiques à rechercher (optionnel)
    my_allergens = ["gluten", "lait", "œufs", "arachides", "soja"]
    
    # Détecter les allergènes
    result = detect_allergens(image_path, my_allergens)
    
    if "error" in result:
        print(f"Erreur: {result['error']}")
    else:
        print("=== RÉSULTATS DE DÉTECTION D'ALLERGÈNES ===")
        print(result["analysis"])
        
    # Pour analyser plusieurs images
    # image_list = ["image1.jpg", "image2.jpg", "image3.jpg"]
    # batch_results = batch_detect_allergens(image_list)