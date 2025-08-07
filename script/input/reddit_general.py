import praw
import pandas as pd
from datetime import datetime
import os
import time

# --- 🔑 Credenciales Reddit ---
reddit = praw.Reddit(
    client_id="_qzeVBBUQWmuuBJeAMEvnw",
    client_secret="AkC9zJm6w8CPbqFfKsJH3Ltr1q2UsQ",
    user_agent="reddit_scraper_r by u/One_Definition_3989"
)


submission = reddit.submission(id="1mbgzdy")
print(submission.title)

# --- Configuración ---
subreddit_name = "chile"
limit_posts = 1000  # máximo permitido por Reddit
output_folder = os.path.expanduser("reddit_data")
os.makedirs(output_folder, exist_ok=True)

# --- Inicializar ---
posts_data = []
comments_data = []

# --- Extraer los posts más recientes ---
print("📥 Descargando posts más recientes...")
subreddit = reddit.subreddit(subreddit_name)
for i, submission in enumerate(subreddit.new(limit=limit_posts)):
    print(f"🔍 [{i+1}/{limit_posts}] {submission.title[:60]}...")

    created_dt = datetime.fromtimestamp(submission.created_utc)

    posts_data.append({
        "id": submission.id,
        "title": submission.title,
        "score": submission.score,
        "author": str(submission.author),
        "num_comments": submission.num_comments,
        "created": created_dt,
        "selftext": submission.selftext,
        "permalink": f"https://reddit.com{submission.permalink}"
    })

    # --- Extraer comentarios ---
    try:
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
        time.sleep(0.5)  # para evitar rate limit
    except Exception as e:
        print(f"⚠️ Error extrayendo comentarios: {e}")


import pandas as pd
import os

# Asegúrate de que estas variables existan:
# posts_data = [...]  # Li_


# --- Guardar Excel ---
output_path = os.path.join(output_folder, "r_chile_reciente.xlsx")
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    pd.DataFrame(posts_data).to_excel(writer, sheet_name="posts", index=False)
    pd.DataFrame(comments_data).to_excel(writer, sheet_name="comentarios", index=False)

print(f"\n✅ Todo guardado en: {output_path}")
