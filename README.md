# Fake News Classification

## Objective

The primary objective of this project is to apply text preprocessing techniques and build a robust machine learning pipeline to classify news articles as either **Fake** or **Real**.

## Dataset

We are using the **Fake News Dataset** for this project.

* **Link to Dataset:** *[[Data](https://drive.google.com/drive/folders/1AS1VVQ6P7CSQJGoHNSb3sQx8bIFSmR9I?usp=drive_link)]*

## Getting Started for the Team

**1. Clone the repository**

```bash
git clone []
cd []
```

**2. Set up a virtual environment (Recommended)**

```bash
python -m venv venv
source venv\Scripts\activate
```

**3. Install dependencies**
Install all required packages (Pandas, Scikit-learn, NLTK, etc.) using the provided requirements file:

```bash
pip install -r requirements.txt
```

**4. Download required NLTK Data**
Since we are performing Natural Language Processing, you will need to download specific NLTK datasets for tokenization and stopword removal. Run the following in your Python environment or at the top of your Jupyter Notebook:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
