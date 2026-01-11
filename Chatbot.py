print("--- TEST DE DÉMARRAGE DU SCRIPT (v6 - MCP Corrigé) ---")
try:
    import requests
    import os
    import json
    print("--- Imports (requests, os, json) OK ---")
except Exception as e:
    print(f"ERREUR PENDANT L'IMPORTATION: {e}")
    exit()

# mcp

def analyse_image(chemin_image_utilisateur):
    url_mcp = "http://127.0.0.1:5000/predict"
    
    if not os.path.exists(chemin_image_utilisateur):
        print(f"[Erreur MCP] Le fichier '{chemin_image_utilisateur}' n'existe pas.")
        return f'{{"erreur": "Le fichier {chemin_image_utilisateur} est introuvable."}}'

    print(f"[MCP] Appel de la fonction 'analyse_image' avec '{chemin_image_utilisateur}'...")

    try:
        with open(chemin_image_utilisateur, 'rb') as f:
            files = {'file': (chemin_image_utilisateur, f, 'image/jpeg')}
            response = requests.post(url_mcp, files=files)
            
            if response.status_code == 200:
                data = response.json()
                print(f"[MCP] Réponse reçue : {data}")
                return response.text 
            else:
                return f'{{"erreur": "Erreur du serveur d\'analyse: {response.text}"}}'
                
    except requests.exceptions.ConnectionError:
        print("[Erreur MCP] Le serveur Flask n'est pas lancé.")
        return '{"erreur": "Erreur critique : Le serveur MCP (Flask) n\'est pas lancé."}'
    except Exception as e:
        return f'{{"erreur": "Une erreur locale est survenue : {e}"}}'


# 2. CONFIGURATION DU LLM prompt

SYSTEM_PROMPT = """
Tu es 'Dr. Feuille', un assistant IA spécialisé dans les maladies des plantes.
- Ton ton est amical et serviable.
- Tu ne dois répondre **uniquement** aux questions sur les plantes et leur santé.
- Si l'utilisateur pose une autre question, refuse poliment en rappelant ta spécialité.

TA TÂCHE 1 (Détection) :
- Si le *dernier message de l'utilisateur* est **juste un nom de fichier** (comme 'image.jpg'), tu ne dois répondre **que** le JSON {"action": "analyse_image", "chemin": "nom_du_fichier.jpg"}. Ne dis rien d'autre.

TA TÂCHE 2 (Formulation) :
- Si le *dernier message de l'utilisateur* contient 'Voici le résultat de l'analyse' et un JSON, tu dois **formuler** ce diagnostic en une belle phrase en français.
- Par exemple, pour '...{"diagnostic": "Tomato___Late_blight"}...', réponds : "Ah, d'après mon analyse, il semble que votre plante souffre de Mildiou (Tomato Late blight)..."

TÂCHE 3 (Conversation) :
- Pour tout autre message texte, réponds normalement à la question.
"""


def appeler_ollama(messages_historique, force_json=False):
    """
    Appelle l'API Ollama manuellement, en contrôlant le format de sortie.
    """
    url_ollama = "http://127.0.0.1:11434/api/chat"
    
    payload = {
    "model": "llama3:instruct",
    "messages": messages_historique,
    "stream": False
      }

    if force_json:
     payload["format"] = "json"

    
    try:
        response = requests.post(url_ollama, json=payload)
        response.raise_for_status()
        return response.json() 
        
    except requests.exceptions.ConnectionError:
        print("\n--- ERREUR CHATBOT ---")
        print("Impossible de contacter Ollama. L'application Ollama est-elle lancée ?")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n--- ERREUR API OLLAMA ---")
        print(f"Erreur : {e}")
        if e.response is not None:
            print(f"Réponse du serveur : {e.response.text}")
        return None

def main_loop():
    print("🤖 Bonjour, je suis Dr. Feuille. Montrez-moi une photo de votre plante (tapez le nom du fichier) ou posez-moi une question.")
    
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT}
    ]

    while True:
        entree_utilisateur = input("Vous : ")
        if not entree_utilisateur:
            continue
        if entree_utilisateur.lower() == "quitter":
            break

        messages.append({'role': 'user', 'content': entree_utilisateur})
        
        # On vérifie si l'utilisateur a envoyé un nom d'image
        is_image_path = entree_utilisateur.lower().endswith((".jpg", ".jpeg", ".png"))

        # --- 1er APPEL (POUR DÉTECTER L'OUTIL OU RÉPONDRE) ---
        
        response_json = appeler_ollama(messages, force_json=is_image_path)
        
        if response_json is None:
            messages.pop() 
            continue
            
        response_message = response_json['message']
        response_text = response_message['content'].strip()
        
        # --- VÉRIFICATION DE LA RÉPONSE JSON ---
        if is_image_path:
            print(f"[LLM] Commande JSON reçue : {response_text}")
            
            try:
                # 1.parse la commande JSON
                data = json.loads(response_text)
                chemin = data['chemin']
                
                # 2.exécute l'outil MCP
                resultat_json_outil = analyse_image(chemin)
                
                # 3.prépare un message pour le 2ème appel
                messages_pour_llm = messages + [
                    {'role': 'assistant', 'content': response_text}, # On ajoute l'appel JSON
                    {'role': 'user', 'content': f"Voici le résultat de l'analyse : {resultat_json_outil}. Formule une belle réponse de diagnostic."}
                ]

                # --- 2ème APPEL (POUR FORMULER LA RÉPONSE) ---
                
                print("[LLM] Envoi du diagnostic à Llama3 pour formulation...")
                final_response_json = appeler_ollama(messages_pour_llm) 
                
                if final_response_json:
                    final_message = final_response_json['message']['content']
                    print(f"Dr. Feuille : {final_message}")
                    messages.append({'role': 'assistant', 'content': final_message})
                
            except (json.JSONDecodeError, KeyError):
                print(f"[Erreur] Le LLM a renvoyé un JSON mal formaté : {response_text}")
                messages.pop()
            except Exception as e:
                print(f"[Erreur] Erreur lors de l'exécution de l'outil : {e}")
                messages.pop()
        
        else:
            
            print(f"Dr. Feuille : {response_text}")
            messages.append({'role': 'assistant', 'content': response_text})


if __name__ == "__main__":
    main_loop()
    