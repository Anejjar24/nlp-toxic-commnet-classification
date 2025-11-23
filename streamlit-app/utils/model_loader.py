# utils/model_loader.py
import tensorflow as tf
import joblib
import os
import logging
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Recréer exactement votre architecture BERT du notebook
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        
        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

class ModelLoader:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.cnn_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.cnn_tokenizer = None
        self.max_length = 150
        self.label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_cnn_model(self):
        """Load CNN model and tokenizer"""
        try:
            # Load CNN model
            cnn_path = os.path.join(self.model_dir, 'cnn_model.h5')
            if os.path.exists(cnn_path):
                self.cnn_model = tf.keras.models.load_model(cnn_path)
                logger.info("✅ CNN model loaded successfully")
            else:
                logger.error(f"❌ CNN model not found at {cnn_path}")
                return False
            
            # Load CNN tokenizer
            tokenizer_path = os.path.join(self.model_dir, 'tokenizers', 'cnn_tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                self.cnn_tokenizer = joblib.load(tokenizer_path)
                logger.info("✅ CNN tokenizer loaded successfully")
                return True
            else:
                logger.error(f"❌ CNN tokenizer not found at {tokenizer_path}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error loading CNN model: {e}")
            return False
    
    def load_bert_model(self):
        """Load BERT model and tokenizer - version personnalisée pour votre modèle .pt"""
        try:
            bert_path = os.path.join(self.model_dir, 'bert_model')
            
            # Vérifier si le modèle BERT existe
            if not os.path.exists(bert_path):
                logger.error(f"❌ BERT model directory not found at {bert_path}")
                return False
            
            # Vérifier que le fichier model_state_dict.pt existe
            model_file = os.path.join(bert_path, 'model_state_dict.pt')
            if not os.path.exists(model_file):
                logger.error(f"❌ BERT model file not found at {model_file}")
                return False
            
            logger.info("🔄 Loading custom BERT model from .pt file...")
            
            # Charger le tokenizer depuis votre dossier BERT
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
                logger.info("✅ BERT tokenizer loaded from local directory")
            except:
                # Fallback au tokenizer de base
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                logger.info("✅ Using default BERT tokenizer")
            
            # Initialiser le modèle avec votre architecture personnalisée
            self.bert_model = BertClassifier(freeze_bert=False)
            
            # Charger les poids depuis votre fichier .pt
            state_dict = torch.load(model_file, map_location=self.device)
            self.bert_model.load_state_dict(state_dict)
            
            # Déplacer le modèle sur le device
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            logger.info("✅ Custom BERT model loaded successfully from .pt file")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading custom BERT model: {e}")
            return False
    
    def preprocess_text_cnn(self, text):
        """Preprocess text for CNN model"""
        from nltk.tokenize import RegexpTokenizer
        from nltk.corpus import stopwords
        import nltk
        
        # Ensure NLTK data is available
        try:
            stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            
        nltk_tokenizer = RegexpTokenizer(r'\w+')
        stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
        
        # Clean the text
        text = text.lower()
        tokens = nltk_tokenizer.tokenize(text)
        filtered = [word for word in tokens if word not in stop_words]
        return " ".join(filtered)
    
    def preprocess_text_bert(self, text):
        """Preprocess text like in your BERT training"""
        import re
        # Reproduire exactement votre fonction de prétraitement BERT
        text = re.sub(r'(@.*?)[\s]', ' ', text)
        text = re.sub(r'[0-9]+' , '' ,text)
        text = re.sub(r'\s([@][\w_-]+)', '', text).strip()
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace("#" , " ")
        encoded_string = text.encode("ascii", "ignore")
        decode_string = encoded_string.decode()
        return decode_string
    
    def predict_cnn_toxicity(self, text):
        """Make prediction using CNN model"""
        if self.cnn_model is None or self.cnn_tokenizer is None:
            raise ValueError("CNN model or tokenizer not loaded")
        
        try:
            # Preprocess text
            cleaned_text = self.preprocess_text_cnn(text)
            
            # Convert to sequences
            sequences = self.cnn_tokenizer.texts_to_sequences([cleaned_text])
            padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post')
            
            # Make prediction
            predictions = self.cnn_model.predict(padded_sequences, verbose=0)
            
            # Format results
            results = {}
            for i, label in enumerate(self.label_names):
                results[label] = float(predictions[0][i])
            
            # Calculate overall toxicity score
            results['overall_toxic'] = max(results.values())
            
            return results
            
        except Exception as e:
            logger.error(f"❌ CNN prediction error: {e}")
            raise
    
    def predict_bert_toxicity(self, text):
        """Make prediction using custom BERT model - Version simplifiée"""
        if self.bert_model is None or self.bert_tokenizer is None:
            raise ValueError("BERT model or tokenizer not loaded")
        
        try:
            processed_text = self.preprocess_text_bert(text)
            
            inputs = self.bert_tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=300
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            with torch.no_grad():
                logits = self.bert_model(input_ids, attention_mask)
                probs = torch.nn.functional.softmax(logits, dim=1)
            
            toxic_prob = probs[0][1].item()
            
            # 🔥 Même logique que votre collègue - EXACTEMENT comme demandé
            results = {
                'toxic': toxic_prob,
                'severe_toxic': toxic_prob * 0.5 if toxic_prob > 0.7 else toxic_prob * 0.2,
                'obscene': toxic_prob * 0.6 if toxic_prob > 0.6 else toxic_prob * 0.3,
                'threat': toxic_prob * 0.3 if toxic_prob > 0.8 else toxic_prob * 0.1,
                'insult': toxic_prob * 0.7 if toxic_prob > 0.5 else toxic_prob * 0.4,
                'identity_hate': toxic_prob * 0.4 if toxic_prob > 0.75 else toxic_prob * 0.15
            }
            
            results['overall_toxic'] = toxic_prob
            
            logger.info(f"🤖 BERT Prediction - Toxic probability: {toxic_prob:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ BERT prediction error: {e}")
            default_results = {label: 0.0 for label in self.label_names}
            default_results['overall_toxic'] = 0.0
            return default_results
    
    def get_models_status(self):
        """Get status of loaded models"""
        return {
            'cnn_loaded': self.cnn_model is not None,
            'bert_loaded': self.bert_model is not None,
            'device': str(self.device)
        }