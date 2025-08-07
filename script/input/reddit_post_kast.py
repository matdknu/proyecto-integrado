import praw
import pandas as pd
from datetime import datetime
import os



# 📌 Conexión con Reddit
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# 📂 Carpeta de salida
downloads_folder = "raw_data"
os.makedirs(downloads_folder, exist_ok=True)

# 📌 Parámetros de búsqueda
subreddit_name = "RepublicadeChile"
query = "Kast"

# 📊 Listas para almacenar datos
posts_data = []
comments_data = []

print(f"🔍 Buscando '{query}' en r/{subreddit_name}...")

# 1️⃣ Buscar posts
for submission in reddit.subreddit(subreddit_name).search(query, sort="new", limit=None):
    # Guardar datos del post
    posts_data.append({
        "id": submission.id,
        "title": submission.title,
        "score": submission.score,
        "author": str(submission.author),
        "num_comments": submission.num_comments,
        "created": datetime.fromtimestamp(submission.created_utc),
        "selftext": submission.selftext,
        "permalink": f"https://reddit.com{submission.permalink}"
    })

    # 2️⃣ Descargar comentarios del post
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        comments_data.append({
            "post_id": submission.id,
            "post_title": submission.title,
            "post_text": submission.selftext,
            "post_url": f"https://reddit.com{submission.permalink}",
            "comment_id": comment.id,
            "author": str(comment.author),
            "body": comment.body,
            "score": comment.score,
            "created": datetime.fromtimestamp(comment.created_utc)
        })

# 3️⃣ Guardar CSV de posts
posts_csv_path = os.path.join(downloads_folder, f"search_posts_{query}.csv")
pd.DataFrame(posts_data).to_csv(posts_csv_path, index=False)

# 4️⃣ Guardar CSV de comentarios
comments_csv_path = os.path.join(downloads_folder, f"search_comments_{query}.csv")
pd.DataFrame(comments_data).to_csv(comments_csv_path, index=False)

# 5️⃣ Mensaje final
print(f"✅ Posts guardados en: {posts_csv_path} ({len(posts_data)} encontrados)")
print(f"✅ Comentarios guardados en: {comments_csv_path} ({len(comments_data)} encontrados)")

