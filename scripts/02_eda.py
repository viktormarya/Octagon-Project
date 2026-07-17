import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter

plt.rcParams['font.family'] = 'DejaVu Sans'  # поддерживает кириллицу
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('../output/reviews_clean.xlsx')
df['Дата'] = pd.to_datetime(df['Дата'])

OUT = '../output'

# ---------- 1. Распределение оценок автора ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

rating_counts = df['Оценка автора'].dropna().value_counts().sort_index()
axes[0].bar(rating_counts.index.astype(str), rating_counts.values, color='#4C72B0')
axes[0].set_title('Распределение оценок отзывов (все источники)')
axes[0].set_xlabel('Оценка (звёзд)')
axes[0].set_ylabel('Число отзывов')
for i, v in enumerate(rating_counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontsize=8)

by_src = df.groupby(['Источник', 'Оценка автора']).size().unstack(fill_value=0)
by_src_pct = by_src.div(by_src.sum(axis=1), axis=0) * 100
by_src_pct.T.plot(kind='bar', ax=axes[1])
axes[1].set_title('Доля оценок по источникам, %')
axes[1].set_xlabel('Оценка (звёзд)')
axes[1].set_ylabel('% отзывов источника')
axes[1].legend(title='Источник')

plt.tight_layout()
plt.savefig(f'{OUT}/01_rating_distribution.png', dpi=130)
plt.close()

print("=== Распределение оценок ===")
print(rating_counts)
print(f"\nСредняя оценка: {df['Оценка автора'].mean():.2f}")
print(f"Медиана: {df['Оценка автора'].median():.1f}")
print(f"Доля 1-2 звезды (негатив): {(df['Оценка автора']<=2).mean()*100:.1f}%")
print(f"Доля 4-5 звёзд (позитив): {(df['Оценка автора']>=4).mean()*100:.1f}%")

# ---------- 2. Динамика по годам ----------
df['Год'] = df['Дата'].dt.year
yearly = df.groupby('Год').agg(отзывов=('Наименование', 'size'), ср_оценка=('Оценка автора', 'mean'))
yearly = yearly[(yearly.index >= 2015) & (yearly.index <= 2024)]

fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.bar(yearly.index.astype(str), yearly['отзывов'], color='#8FA8D6', label='Число отзывов')
ax1.set_ylabel('Число отзывов')
ax2 = ax1.twinx()
ax2.plot(yearly.index.astype(str), yearly['ср_оценка'], color='#C44E52', marker='o', label='Средняя оценка')
ax2.set_ylabel('Средняя оценка')
ax2.set_ylim(1, 5)
ax1.set_title('Число отзывов и средняя оценка по годам')
fig.legend(loc='upper left', bbox_to_anchor=(0.08, 0.88))
plt.tight_layout()
plt.savefig(f'{OUT}/02_yearly_trend.png', dpi=130)
plt.close()

print("\n=== По годам ===")
print(yearly.round(2))

# ---------- 3. Длина отзыва vs оценка ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
data_by_rating = [df.loc[df['Оценка автора'] == r, 'Длина_отзыва_слов'].dropna() for r in [1, 2, 3, 4, 5]]
ax.boxplot(
    data_by_rating,
    tick_labels=['1', '2', '3', '4', '5'],
    showfliers=False
)
ax.set_xlabel('Оценка (звёзд)')
ax.set_ylabel('Длина отзыва, слов')
ax.set_title('Длина отзыва в зависимости от оценки')
plt.tight_layout()
plt.savefig(f'{OUT}/03_length_vs_rating.png', dpi=130)
plt.close()

print("\n=== Средняя длина отзыва по оценке ===")
print(df.groupby('Оценка автора')['Длина_отзыва_слов'].agg(['mean', 'median', 'count']).round(1))

# ---------- 4. Топ компаний ----------
top_companies = df.groupby('Наименование').agg(
    отзывов=('Наименование', 'size'),
    ср_оценка=('Оценка автора', 'mean')
).query('отзывов >= 30').sort_values('отзывов', ascending=False).head(15)
print("\n=== Топ-15 компаний по числу отзывов (>=30 отзывов) ===")
print(top_companies.round(2))

worst_companies = df.groupby('Наименование').agg(
    отзывов=('Наименование', 'size'),
    ср_оценка=('Оценка автора', 'mean')
).query('отзывов >= 30').sort_values('ср_оценка').head(10)
print("\n=== Компании с самой низкой средней оценкой (>=30 отзывов) ===")
print(worst_companies.round(2))

# ---------- 5. Частотный анализ слов: позитив vs негатив ----------
STOPWORDS = set("""
и в не на я что тот быть с а весь это как он но они мы вы к у же вот для по от до
из о ли бы то за при из-за или всё все был было была были этот эта эти него нее
свой который мой наш ваш их ее его если чтобы когда где там тут здесь очень
уже еще ещё так такой такая такие можно нужно надо просто только тоже во со ко
да нет ни ну да вообще типа как-то что-то кто-то один одна одно два три
себя себе меня мне тебя тебе нам вам им ему ей нас вас них него неё
быть был будет были будем будете тем те того той том чем кем некоторые
после перед над под между через без благодаря вместо кроме около
мебель магазин заказ заказал заказала заказали купил купила купили
компания компании фирма салон место продукт продукция товар вещь
всё это тот та то этот эта год года лет назад месяц месяцев
""".split())

def tokenize(text):
    words = re.findall(r'[а-яё]{3,}', str(text).lower())
    return [w for w in words if w not in STOPWORDS]

neg_texts = df.loc[(df['Оценка автора'] <= 2) & df['Есть_текст'], 'Текст']
pos_texts = df.loc[(df['Оценка автора'] >= 4) & df['Есть_текст'], 'Текст']

neg_words = Counter()
for t in neg_texts:
    neg_words.update(tokenize(t))

pos_words = Counter()
for t in pos_texts:
    pos_words.update(tokenize(t))

print(f"\n=== Топ-25 слов в НЕГАТИВНЫХ отзывах (1-2 звезды, n={len(neg_texts)}) ===")
for w, c in neg_words.most_common(25):
    print(f"  {w}: {c}")

print(f"\n=== Топ-25 слов в ПОЗИТИВНЫХ отзывах (4-5 звёзд, n={len(pos_texts)}) ===")
for w, c in pos_words.most_common(25):
    print(f"  {w}: {c}")

# график топ слов негатива
fig, ax = plt.subplots(figsize=(8, 7))
top_neg = neg_words.most_common(20)[::-1]
ax.barh([w for w, c in top_neg], [c for w, c in top_neg], color='#C44E52')
ax.set_title('Топ-20 слов в негативных отзывах (1-2 звезды)')
plt.tight_layout()
plt.savefig(f'{OUT}/04_top_negative_words.png', dpi=130)
plt.close()

# сохраняем словари частот для дальнейшей категоризации жалоб
import json
with open(f'{OUT}/word_freq_negative.json', 'w', encoding='utf-8') as f:
    json.dump(neg_words.most_common(200), f, ensure_ascii=False, indent=2)
with open(f'{OUT}/word_freq_positive.json', 'w', encoding='utf-8') as f:
    json.dump(pos_words.most_common(200), f, ensure_ascii=False, indent=2)

print("\nГрафики сохранены в", OUT)
