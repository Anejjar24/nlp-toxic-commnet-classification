# test_bert.py
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Recréer votre architecture exacte
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

def test_my_bert_model():
    bert_path = 'models/bert_model'
    
    print("🔍 Testing your custom BERT model...")
    print(f"BERT path: {bert_path}")
    
    # Vérifier les fichiers
    if os.path.exists(bert_path):
        files = os.listdir(bert_path)
        print(f"📁 Files in BERT directory: {files}")
        
        # Essayer de charger le modèle
        try:
            print("🔄 Loading tokenizer...")
            tokenizer = BertTokenizer.from_pretrained(bert_path)
            print("✅ Tokenizer loaded successfully!")
            
            print("🔄 Loading model...")
            model_file = os.path.join(bert_path, 'model_state_dict.pt')
            
            # Initialiser le modèle
            model = BertClassifier()
            
            # Charger les poids
            state_dict = torch.load(model_file, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            
            print("✅ Model loaded successfully!")
            
            # Tester une prédiction
            print("🧪 Testing prediction...")
            test_texts = [
                "This is a friendly comment, thank you!",
                "You are stupid and I hate you!",
                "This is fucking ridiculous",
                "I appreciate your help"
            ]
            
            for test_text in test_texts:
                print(f"\n📝 Text: '{test_text}'")
                
                # Prétraitement comme dans votre notebook
                import re
                def text_preprocessing(text):
                    text = re.sub(r'(@.*?)[\s]', ' ', text)
                    text = re.sub(r'[0-9]+' , '' ,text)
                    text = re.sub(r'\s([@][\w_-]+)', '', text).strip()
                    text = re.sub(r'&amp;', '&', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    text = text.replace("#" , " ")
                    encoded_string = text.encode("ascii", "ignore")
                    decode_string = encoded_string.decode()
                    return decode_string
                
                processed_text = text_preprocessing(test_text)
                print(f"🔧 Processed: '{processed_text}'")
                
                # Tokenizer
                inputs = tokenizer(
                    processed_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=300
                )
                
                # Extraire seulement input_ids et attention_mask
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                
                with torch.no_grad():
                    logits = model(input_ids, attention_mask)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                
                toxic_prob = probs[0][1].item()
                non_toxic_prob = probs[0][0].item()
                
                print(f"📊 Predictions:")
                print(f"  Non-toxic: {non_toxic_prob:.4f}")
                print(f"  Toxic: {toxic_prob:.4f}")
                print(f"  Classification: {'🚨 TOXIC' if toxic_prob > 0.5 else '✅ CLEAN'}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("❌ BERT directory not found!")

if __name__ == "__main__":
    test_my_bert_model()