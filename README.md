# Toxic Comment Classification -- Deep Learning Project

## 📌 Overview

This project focuses on **multi-label toxic comment classification**,
aiming to automatically detect several categories of online toxicity,
including:

-   Toxic
-   Severe toxic
-   Obscene
-   Threat
-   Insult
-   Identity hate

The repository contains:

-   Deep Learning notebooks (**CNN**, **BERT**, **EDA**)
-   A fully functional **Streamlit application**
-   A **PDF report**
-   A **presentation PPT** summarizing the work

------------------------------------------------------------------------

## 🔬 Models & Approaches

### **1. CNN with FastText Embeddings**

-   Uses **FastText word embeddings**
-   Multi-scale convolutional filters
-   Captures local text patterns efficiently
-   Fast training and good performance across labels
-   Implemented with **TensorFlow/Keras**

### **2. BERT Transformer Model**

-   Contextualized bidirectional encoding
-   Captures complex semantics and long dependencies
-   More accurate than CNN but more computationally expensive
-   Implemented using **Hugging Face Transformers** and **PyTorch**

------------------------------------------------------------------------

## 📊 Exploratory Data Analysis (EDA)

The **Toxic_Comments_EDA.ipynb** notebook includes:

-   Label distributions
-   Word frequency analysis
-   Comment length distributions
-   Examples of toxic vs. non-toxic comments

------------------------------------------------------------------------

## 🧰 Tools & Libraries

### **Deep Learning**

-   PyTorch
-   TensorFlow / Keras
-   Hugging Face Transformers

### **NLP**

-   NLTK
-   SpaCy
-   FastText

### **Deployment**

-   Streamlit
-   Matplotlib & Seaborn

------------------------------------------------------------------------

## 📁 Project Structure

    Final-NLP-Toxic-Comments-Classification/
    │
    ├── Bert_Model.ipynb
    ├── Keras_CNN_with_FastText.ipynb
    ├── Toxic_Comments_EDA.ipynb
    │
    ├── streamlit-app/
    │   ├── app.py
    │   ├── requirements.txt
    │   └── assets/
    │
    ├── report/
    │   └── Toxic_Comments_Classification_Report.pdf
    │
    ├── presentation/
    │   └── Toxic_Comments_Classification_Presentation.pptx
    │
    └── README.md

------------------------------------------------------------------------

## 🚀 Run the Streamlit App

### 1. Install Dependencies

``` bash
pip install -r streamlit-app/requirements.txt
```

### 2. Launch the App

``` bash
streamlit run streamlit-app/app.py
```

------------------------------------------------------------------------

## 📄 Report & Presentation

-   Full PDF report: `report/Toxic_Comments_Classification_Report.pdf`
-   Project presentation:
    `presentation/Toxic_Comments_Classification_Presentation.pptx`

------------------------------------------------------------------------

## 👤 Author

**Ihsane** Machine Learning & NLP Enthusiast
