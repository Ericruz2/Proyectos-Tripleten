import math

import numpy as np
import pandas as pd
from IPython.display import display

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import re

from tqdm.auto import tqdm


import matplotlib.pyplot as plt
print(plt.style.available)

# esto es para usar progress_apply, puedes leer más en https://pypi.org/project/tqdm/#pandas-integration
tqdm.pandas()

df_reviews = pd.read_csv('C:/Users/agustin/Downloads/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})

print (df_reviews.info())
print (df_reviews.head())
print (df_reviews['pos'].value_counts())
print (df_reviews['review'])

# Número de películas por año
dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates()['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)

print("-- Películas por año (dft1):")
print(dft1)

# Reseñas por año y polaridad
dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack().fillna(0)
print("\n --> Reseñas por año y polaridad (dft2 - stacked):")
print(dft2)

# Total de reseñas por año
dft2_total = df_reviews['start_year'].value_counts().sort_index()
dft2_total = dft2_total.reindex(index=np.arange(dft2_total.index.min(), max(dft2_total.index.max(), 2021))).fillna(0)
print("\n -- Total de reseñas por año (dft2 - total):")
print(dft2_total)

# Promedio de reseñas por película
dft3 = (dft2_total / dft1).fillna(0)
print("\n -- Promedio de reseñas por película (dft3):")
print(dft3)

# Resumen estadístico
dft = df_reviews.groupby('tconst')['review'].count()
print(dft.describe())  

import sklearn.metrics as metrics
def evaluate_model(model, train_features, train_target, test_features, test_target):

    eval_stats = {}

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):

        eval_stats[type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]

        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps

        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1')

        # ROC
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')
        ax.set_title(f'Curva ROC')

        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)

    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))

    print(df_eval_stats)

    return df_eval_stats, fig

# Normalización de reseñas: minúsculas + eliminar dígitos y signos
def normalize_text(text):
    return re.sub(r'[^a-z\s]', '', text.lower())

df_reviews['review_norm'] = df_reviews['review'].apply(normalize_text) # <escribe tu código aquí>

print (df_reviews['review'])


df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

print(df_reviews_train.shape)
print(df_reviews_test.shape)

# --- Librerías ---
import numpy as np
import pandas as pd
import spacy

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

# --- Función de evaluación ---
def evaluar_modelo(nombre, modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    train_preds = modelo.predict(X_train)
    test_preds = modelo.predict(X_test)

    resultados = {
        "Modelo": nombre,
        "Train Accuracy": accuracy_score(y_train, train_preds),
        "Train F1": f1_score(y_train, train_preds),
        "Test Accuracy": accuracy_score(y_test, test_preds),
        "Test F1": f1_score(y_test, test_preds)
    }
    return resultados


# ==============================
# Modelo 0 - Constante
# ==============================
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
res_dummy = evaluar_modelo(
    "Constante",
    dummy,
    np.zeros((len(train_target), 1)), train_target,
    np.zeros((len(test_target), 1)), test_target
)
print(res_dummy)


# ==============================
# Modelo 1 - NLTK + TF-IDF + LR
# ==============================
vectorizer_nltk = TfidfVectorizer()
X_train_nltk = vectorizer_nltk.fit_transform(df_reviews_train["review_norm"])
X_test_nltk = vectorizer_nltk.transform(df_reviews_test["review_norm"])

logreg_nltk = LogisticRegression(random_state=42, max_iter=500)
res_logreg_nltk = evaluar_modelo(
    "NLTK + TF-IDF + LR",
    logreg_nltk,
    X_train_nltk, train_target,
    X_test_nltk, test_target
)
print(res_logreg_nltk)


# ==============================
# Modelo 3 - spaCy + TF-IDF + LR
# ==============================
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def text_preprocessing_spacy(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

df_reviews_train["review_spacy"] = df_reviews_train["review"].apply(text_preprocessing_spacy)
df_reviews_test["review_spacy"] = df_reviews_test["review"].apply(text_preprocessing_spacy)

vectorizer_spacy = TfidfVectorizer()
X_train_spacy = vectorizer_spacy.fit_transform(df_reviews_train["review_spacy"])
X_test_spacy = vectorizer_spacy.transform(df_reviews_test["review_spacy"])

logreg_spacy = LogisticRegression(random_state=42, max_iter=500)
res_logreg_spacy = evaluar_modelo(
    "spaCy + TF-IDF + LR",
    logreg_spacy,
    X_train_spacy, train_target,
    X_test_spacy, test_target
)
print(res_logreg_spacy)


# ==============================
# Modelo 4 - spaCy + TF-IDF + LGBMClassifier
# ==============================
lgbm = LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.1)
res_lgbm = evaluar_modelo(
    "spaCy + TF-IDF + LGBMClassifier",
    lgbm,
    X_train_spacy, train_target,
    X_test_spacy, test_target
)
print(res_lgbm)


# ----- 




# ----- 




# ----- 