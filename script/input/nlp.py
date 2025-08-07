# ========================================
# 1️⃣ IMPORTAR LIBRERÍAS
# ========================================
import pandas as pd
import spacy
import re
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pysentimiento import create_analyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
import pyLDAvis

# ========================================
# 2️⃣ CARGA Y LIMPIEZA DE DATOS
# ========================================
df = pd.read_csv("raw_data/search_posts_Kast.csv")

# Cargar modelo SpaCy y stopwords
nlp = spacy.load("es_core_news_sm")
stop_words = set(stopwords.words("spanish"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Quitar URLs
    text = re.sub(r"[^a-záéíóúüñ\s]", '', text)        # Solo letras
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

df["text_clean"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).apply(clean_text)

# ========================================
# 3️⃣ EXPLORACIÓN: FRECUENCIAS Y SENTIMIENTO
# ========================================
# Top 20 palabras más frecuentes
word_counts = Counter(" ".join(df["text_clean"]).split())
freq_df = pd.DataFrame(word_counts.items(), columns=["Palabra", "Frecuencia"]).sort_values(by="Frecuencia", ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(freq_df["Palabra"].head(20), freq_df["Frecuencia"].head(20))
plt.xticks(rotation=45)
plt.title("Top 20 palabras más frecuentes")
plt.show()

# Nube de palabras
wc = WordCloud(width=800, height=400, background_color="white", colormap="Dark2").generate(" ".join(df["text_clean"]))
plt.imshow(wc, interpolation="bilinear"); plt.axis("off"); plt.show()

# Análisis de sentimiento
analyzer = create_analyzer(task="sentiment", lang="es")
df["sentiment"] = df["text_clean"].apply(lambda x: analyzer.predict(x).output)
df["sentiment"].value_counts().plot(kind="bar", color=["green", "red", "gray"], title="Distribución de Sentimientos")
plt.show()

# ========================================
# 4️⃣ TOPIC MODELING (LDA) Y 2 TEMAS
# ========================================
vectorizer = CountVectorizer(max_df=0.8, min_df=5)
X_counts = vectorizer.fit_transform(df["text_clean"])

# Selección de número de temas por perplejidad
perplexities = []
for k in range(2, 7):
    lda_k = LatentDirichletAllocation(n_components=k, random_state=42)
    lda_k.fit(X_counts)
    perplexities.append(lda_k.perplexity(X_counts))
plt.plot(range(2, 7), perplexities, marker="o"); plt.xlabel("Temas"); plt.ylabel("Perplejidad"); plt.show()

# LDA con 2 temas
lda_2 = LatentDirichletAllocation(n_components=2, random_state=42)
lda_2.fit(X_counts)
terms = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda_2.components_):
    print(f"Tema {idx}: {', '.join([terms[i] for i in topic.argsort()[-10:]])}")

# Asignar tema dominante
topic_probs = lda_2.transform(X_counts)
df["tema_dominante"] = topic_probs.argmax(axis=1)
df["confianza_tema"] = topic_probs.max(axis=1)

# Etiquetas semánticas
mapa_temas = {0: "Debate político general y discusiones internas", 1: "Estrategia electoral y liderazgos de la derecha"}
df["etiqueta_tema"] = df["tema_dominante"].map(mapa_temas)

# ========================================
# 5️⃣ CRUCE SENTIMIENTO × TEMA Y EVOLUCIÓN
# ========================================
# Cruce de variables
cruce = pd.crosstab(df["etiqueta_tema"], df["sentiment"], normalize="index")*100
cruce.plot(kind="barh", stacked=True, color={"POS":"green", "NEG":"red", "NEU":"gray"})
plt.title("Sentimiento por Tema"); plt.show()

# Evolución temporal
df["created"] = pd.to_datetime(df["created"], errors="coerce")
df = df.dropna(subset=["created"])
df["mes"] = df["created"].dt.to_period("M")
evolucion = df.groupby(["mes", "etiqueta_tema"]).size().reset_index(name="conteo")
for tema in evolucion["etiqueta_tema"].unique():
    sub = evolucion[evolucion["etiqueta_tema"] == tema]
    plt.plot(sub["mes"].astype(str), sub["conteo"], marker="o", label=tema)
plt.xticks(rotation=45); plt.legend(); plt.title("Evolución de temas"); plt.show()

# ========================================
# 6️⃣ MODELOS SUPERVISADOS Y COMPARACIÓN
# ========================================
X = df["text_clean"]
y = df["etiqueta_tema"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def entrenar_y_evaluar(nombre, modelo):
    pipeline = Pipeline([("tfidf", TfidfVectorizer(max_df=0.8, min_df=3, ngram_range=(1,2))), ("clf", modelo)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"\n### {nombre} ###\n", classification_report(y_test, y_pred))
    cv_score = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"Accuracy CV (5 folds): {cv_score.mean():.3f}")
    cm = confusion_matrix(y_test, y_pred, labels=y.unique())
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=y.unique(), yticklabels=y.unique()); plt.show()
    # ROC si aplica
    if hasattr(modelo, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test.map({y.unique()[0]:0, y.unique()[1]:1}), y_prob)
        fpr, tpr, _ = roc_curve(y_test.map({y.unique()[0]:0, y.unique()[1]:1}), y_prob)
        plt.plot(fpr, tpr, label=f"{nombre} AUC={auc:.2f}"); plt.plot([0,1],[0,1],"--"); plt.legend(); plt.show()

modelos = [
    ("SVM", LinearSVC()),
    ("Regresión Logística", LogisticRegression(max_iter=1000)),
    ("SGDClassifier", SGDClassifier(loss="log_loss", max_iter=1000)),
    ("Naive Bayes", MultinomialNB()),
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42))
]

for nombre, modelo in modelos:
    entrenar_y_evaluar(nombre, modelo)

# ========================================
# 7️⃣ PREDICCIÓN EN NUEVOS DATOS
# ========================================
pipeline_final = Pipeline([
    ("tfidf", TfidfVectorizer(max_df=0.8, min_df=3, ngram_range=(1,2))),
    ("clf", LinearSVC())
])
pipeline_final.fit(X, y)

nuevos_posts = [
    "Matthei anuncia candidatura presidencial y genera debate en la derecha",
    "La delincuencia en las calles sigue aumentando según la ciudadanía",
]
preds = pipeline_final.predict(nuevos_posts)
for post, pred in zip(nuevos_posts, preds):
    print(f"Post: {post} → Tema: {pred}")
