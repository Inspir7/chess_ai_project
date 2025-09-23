import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Свързване с базата данни
conn = sqlite3.connect("datachess_games.db")
cursor = conn.cursor()

# Брой игри
cursor.execute("SELECT COUNT(*) FROM games")
game_count = cursor.fetchone()[0]
print(f"Общо игри в базата: {game_count}")

# Среден брой ходове
cursor.execute("SELECT AVG(num_moves) FROM games")
avg_moves = cursor.fetchone()[0]
print(f"Среден брой ходове в игрите: {avg_moves:.2f}")

# Най-често срещани дебюти
query = """
SELECT opening, COUNT(*) as count FROM games
GROUP BY opening
ORDER BY count DESC
LIMIT 10
"""
openings_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 5))
sns.barplot(x="count", y="opening", data=openings_df, legend=False, palette="viridis")
plt.xlabel("Брой игри")
plt.ylabel("Дебют")
plt.title("Най-популярните дебюти")
plt.show()

# Анализ на резултатите
query = """
SELECT result, COUNT(*) as count FROM games
GROUP BY result
"""
results_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(6, 4))
sns.barplot(x="result", y="count", data=results_df, legend=False, palette="coolwarm")
plt.xlabel("Резултат")
plt.ylabel("Брой игри")
plt.title("Разпределение на резултатите")
plt.show()

# Дебюти, водещи до най-много победи за белите
query = """
SELECT opening, COUNT(*) as count FROM games
WHERE result = '1-0'
GROUP BY opening
ORDER BY count DESC
LIMIT 10
"""
white_wins_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 5))
sns.barplot(x="count", y="opening", data=white_wins_df, legend=False, palette="Blues")
plt.xlabel("Брой победи за белите")
plt.ylabel("Дебют")
plt.title("Дебюти, водещи до най-много победи за белите")
plt.show()

# Дебюти, водещи до най-много победи за черните
query = """
SELECT opening, COUNT(*) as count FROM games
WHERE result = '0-1'
GROUP BY opening
ORDER BY count DESC
LIMIT 10
"""
black_wins_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 5))
sns.barplot(x="count", y="opening", data=black_wins_df, legend=False, palette="Reds")
plt.xlabel("Брой победи за черните")
plt.ylabel("Дебют")
plt.title("Дебюти, водещи до най-много победи за черните")
plt.show()

# Разлика в броя ходове между различните дебюти
query = """
SELECT opening, AVG(num_moves) as avg_moves FROM games
GROUP BY opening
ORDER BY avg_moves DESC
LIMIT 10
"""
moves_per_opening_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 5))
sns.barplot(x="avg_moves", y="opening", data=moves_per_opening_df, legend=False, palette="magma")
plt.xlabel("Среден брой ходове")
plt.ylabel("Дебют")
plt.title("Среден брой ходове за различните дебюти")
plt.show()

# Дебюти, които водят най-често до реми
query = """
SELECT opening, COUNT(*) as count FROM games
WHERE result = '1/2-1/2'
GROUP BY opening
ORDER BY count DESC
LIMIT 10
"""
draw_openings_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 5))
sns.barplot(x="count", y="opening", data=draw_openings_df, legend=False, palette="Greens")
plt.xlabel("Брой ремита")
plt.ylabel("Дебют")
plt.title("Дебюти, които водят най-често до реми")
plt.show()

# Анализ на личните игри на inspresi
query = """
SELECT result, COUNT(*) as count FROM games WHERE white = 'inspresi' OR black = 'inspresi' GROUP BY result
"""
user_results_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(6, 4))
sns.barplot(x="result", y="count", data=user_results_df, legend=False, palette="coolwarm")
plt.xlabel("Резултат")
plt.ylabel("Брой игри")
plt.title("Резултати от игрите на inspresi")
plt.show()

query = """
SELECT opening, COUNT(*) as count FROM games WHERE white = 'inspresi' OR black = 'inspresi' GROUP BY opening ORDER BY count DESC LIMIT 10
"""
user_openings_df = pd.read_sql_query(query, conn)

plt.figure(figsize=(10, 5))
sns.barplot(x="count", y="opening", data=user_openings_df, legend=False, palette="Purples")
plt.xlabel("Брой игри")
plt.ylabel("Дебют")
plt.title("Най-често използвани дебюти от inspresi")
plt.show()

# Затваряне на връзката с базата
conn.close()
