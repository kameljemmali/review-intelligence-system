import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

category_mapping = {
    'U.S. NEWS': 'News & Politics',
    'WORLD NEWS': 'News & Politics',
    'POLITICS': 'News & Politics',
    'THE WORLDPOST': 'News & Politics',
    'WORLDPOST': 'News & Politics',
    'CRIME' : 'News & Politics',

    'ENTERTAINMENT': 'Entertainment',
    'COMEDY': 'Entertainment',
    'WEIRD NEWS': 'Entertainment',

    'PARENTING': 'Family',
    'PARENTS': 'Family',
    'WEDDINGS': 'Family',
    'DIVORCE': 'Family',

    'ARTS': 'Arts & Culture',
    'ARTS & CULTURE': 'Arts & Culture',
    'CULTURE & ARTS': 'Arts & Culture',

    'BUSINESS': 'Business',
    'MONEY': 'Business',

    'SCIENCE': 'Science & Tech',
    'TECH': 'Science & Tech',

    'ENVIRONMENT': 'Environment',
    'GREEN': 'Environment',

    'WELLNESS': 'Health',
    'HEALTHY LIVING': 'Health',

    'FOOD & DRINK': 'Lifestyle',
    'STYLE': 'Lifestyle',
    'STYLE & BEAUTY': 'Lifestyle',
    'HOME & LIVING': 'Lifestyle',
    'TRAVEL': 'Lifestyle',
    'TASTE': 'Lifestyle',

    'SPORTS': 'Sports',

    'QUEER VOICES': 'identity',
    'BLACK VOICES': 'identity',
    'LATINO VOICES': 'identity',
    'WOMEN': 'identity',


    'EDUCATION': 'Education',
    'COLLEGE': 'Education',

    'MEDIA': 'Media & Impact',
    'IMPACT': 'Media & Impact',

    'RELIGION': 'Religion',

    'GOOD NEWS': 'Good News',
    'FIFTY': 'Good News'


}



def clean_text(text:str) -> str:
    """
    Cleans the input text by removing HTML tags, special characters, and extra whitespace.

    Args:
        text (str): The input text to be cleaned."""
    
    if pd.isna(text):
        return "" #si le texte est une valeur manquante la fonction retourne une chaîne vide ""
    text=str(text).lower()#Convertit le texte en chaîne de caractères puis met le texte en miniscule
    text=re.sub(r"http\S+|www\S+|https\S+", '', text) # supprime les liens internets
    text=re.sub(r"[^a-zA-Z0-9\s]", '', text)#supprimer les caracteres spéciaux on garde seulement les lettres, les chiffres et les espaces
    text=re.sub(r"\s+", ' ', text).strip()#remplacer plusieurs espaces par un seul espace
    return text



def prepare_dataset(df:pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    missing_values = ["", " ", "NA", "N/A", "na", "n/a", "?", "-"]
    df.replace(missing_values, pd.NA, inplace=True)
    df.drop_duplicates(inplace=True)


    df["authors"]=df["authors"].fillna("Unknown")
    df["headline"]=df["headline"].fillna("")
    df["short_description"]=df["short_description"].fillna(df["headline"])
    

    df.dropna(subset=["category"], inplace=True)
    df["category_original"]=df["category"]
    df["category"]=df["category"].map(category_mapping)


    df.dropna(subset=["category"], inplace=True)
    df["text"]=df["headline"]+" "+df["short_description"]
    df["cleaned_text"]=df["text"].apply(clean_text)

    df["text_length"]=df["text"].apply(len)
    df["word_count"]=df["text"].apply(lambda x: len(x.split()))

    df=process_date(df)
    df,encoder=encode_labels(df)
    return df,encoder


def process_date(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df



def encode_labels(df:pd.DataFrame) :
    encoder=LabelEncoder()
    df=df.copy()
    df["category_encoded"]=encoder.fit_transform(df["category"])
    return df,encoder
    