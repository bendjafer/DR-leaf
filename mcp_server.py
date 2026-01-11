import io
import json
import torch
import torch.nn as nn # <-- Importez nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from collections import OrderedDict



# Charger les noms des classes
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"{len(class_names)} classes.")

# Transformations
transformation_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Transformations d'image chargées.")


import torchvision.models as models
modele_specialiste = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)



num_classes = len(class_names)
num_ftrs = modele_specialiste.classifier[1].in_features

modele_specialiste.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),

    nn.Dropout(p=0.5),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),

    nn.Dropout(p=0.5),
    nn.Linear(256, num_classes)
)



# 4. Charger les Weights
try:
    checkpoint = torch.load(
        'best_plantdoc_model.pth', 
        map_location=torch.device('cpu')
    )
    
    state_dict = checkpoint['model_state_dict']
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('mobilenet.', '', 1) 
        new_state_dict[name] = v

    # On charge le dictionnaire nettoyé
    modele_specialiste.load_state_dict(new_state_dict)
    
    modele_specialiste.eval()
    print("---------------------------------------------------------")
    print("Poids 'best_plant_disease_model.pth' chargés AVEC SUCCÈS.")
    print("---------------------------------------------------------")

except RuntimeError as e:
    print("ERREUR LORS DU CHARGEMENT DES POIDS")
    print(f"Erreur: {e}")
    exit()
except Exception as e:
    print(f"Erreur inconnue : {e}")
    exit()

#  FLASK 

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predire_maladie():
    if 'file' not in request.files:
        return jsonify({'erreur': 'Aucun fichier trouvé'}), 400
    
    fichier_image = request.files['file']
    
    try:
        image_bytes = fichier_image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        image_tensor = transformation_image(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = modele_specialiste(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, 1)
            
          
            predicted_idx = top_idx.item() 
            predicted_class_name = class_names[predicted_idx]
            confidence = top_prob.item()

        return jsonify({
            'diagnostic': predicted_class_name,
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'erreur': str(e)}), 500

if __name__ == '__main__':
    print("Serveur MCP démarré. Attente de requêtes sur http://127.0.0.1:5000/predict") # Note : J'ai mis 127.0.0.1, mais 127.0.0.1 est standard
    app.run(debug=True, port=5000)