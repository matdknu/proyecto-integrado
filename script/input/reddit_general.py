import praw
import pandas as pd
from datetime import datetime
import os



# 📌 Conexión con Reddit
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# 📌 ID del post
post_id = "1mbgzdy"
submission = reddit.submission(id=post_id)

# 📂 Carpeta Descargas
downloads_folder = os.path.expanduser("raw_data")

# 1️⃣ Datos del post
post_data = [{
    "id": submission.id,
    "title": submission.title,
    "score": submission.score,
    "author": str(submission.author),
    "num_comments": submission.num_comments,
    "created": datetime.fromtimestamp(submission.created_utc),
    "selftext": submission.selftext,
    "permalink": f"https://reddit.com{submission.permalink}"
}]

# Guardar CSV del post
post_csv_path = os.path.join(downloads_folder, f"post_{post_id}.csv")
pd.DataFrame(post_data).to_csv(post_csv_path, index=False)

# 2️⃣ Descargar comentarios
submission.comments.replace_more(limit=0)
comments_data = []
for comment in submission.comments.list():
    comments_data.append({
        "post_id": submission.id,
        "post_title": submission.title,  # ✅ Título del post
        "post_text": submission.selftext,  # ✅ Texto completo del post
        "post_url": f"https://reddit.com{submission.permalink}",  # ✅ URL del post
        "comment_id": comment.id,
        "author": str(comment.author),
        "body": comment.body,
        "score": comment.score,
        "created": datetime.fromtimestamp(comment.created_utc)
    })


# Guardar CSV de comentarios
comments_csv_path = os.path.join(downloads_folder, f"comments_{post_id}.csv")
pd.DataFrame(comments_data).to_csv(comments_csv_path, index=False)

# 3️⃣ Mensaje final
print(f"✅ Post guardado en: {post_csv_path}")
print(f"✅ Comentarios guardados en: {comments_csv_path}")
print(f"📊 Total comentarios: {len(comments_data)}")
