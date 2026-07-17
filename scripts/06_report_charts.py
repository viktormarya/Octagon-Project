import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('../output/reviews_categorized.xlsx')
df['Текст_lower'] = df['Текст'].fillna('').str.lower()
OUT = '../output'

CATS = ['Доставка/сроки', 'Качество товара', 'Сервис/персонал', 'Цена/деньги/возврат', 'Несоответствие описанию']
overall = pd.Series({c: df[f'Жалоба_{c}'].sum() for c in CATS}).sort_values()
neg = df[df['Оценка автора'] <= 2]
neg_counts = pd.Series({c: neg[f'Жалоба_{c}'].sum() for c in CATS})[overall.index]

fig, ax = plt.subplots(figsize=(9, 4.5))
y = np.arange(len(CATS))
h = 0.36
ax.barh(y - h/2, overall.values, height=h, label='Все отзывы с текстом (n=%d)' % df['Есть_текст'].sum(), color='#A9724C')
ax.barh(y + h/2, neg_counts.values, height=h, label='Только негативные 1-2★ (n=%d)' % len(neg), color='#A6453A')
ax.set_yticks(y); ax.set_yticklabels(overall.index)
ax.set_xlabel('Число отзывов, где встречается категория')
ax.set_title('Гипотеза 3: частота категорий жалоб')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/05_complaints_categories.png', dpi=130)
plt.close()

# H2: fast vs slow delivery rating
FAST = re.compile(r'быстро.{0,40}(доставили|привезли|доставк)|доставили.{0,10}быстро|привезли.{0,15}(быстро|раньше срока|раньше)')
SLOW = re.compile(r'долго.{0,40}(доставк|привезл|жда)|доставк\w*.{0,20}(задерж|перенос|опозда)|опозда|задерж\w*.{0,15}(доставк|привоз)|не привезли вовремя')
df['fast'] = df['Текст_lower'].apply(lambda t: bool(FAST.search(t)))
df['slow'] = df['Текст_lower'].apply(lambda t: bool(SLOW.search(t)))
fast_r = df.loc[df['fast'] & ~df['slow'], 'Оценка автора'].dropna()
slow_r = df.loc[df['slow'] & ~df['fast'], 'Оценка автора'].dropna()

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(['Быстрая доставка\n(n=%d)' % len(fast_r), 'Медленная/сорванная\n(n=%d)' % len(slow_r)],
               [fast_r.mean(), slow_r.mean()], color=['#5C7A5C', '#A6453A'], width=0.5)
ax.set_ylim(0, 5)
ax.set_ylabel('Средняя оценка')
ax.set_title('Гипотеза 2: скорость доставки и оценка')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.08, f'{b.get_height():.2f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/06_delivery_speed.png', dpi=130)
plt.close()

# H4: soft vs corpus furniture
soft = df[(df['Тип_мебели'] == 'Мягкая') & df['Есть_текст']]
corpus = df[(df['Тип_мебели'] == 'Корпусная') & df['Есть_текст']]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].bar(['Мягкая\n(n=%d)' % len(soft), 'Корпусная\n(n=%d)' % len(corpus)],
            [soft['Длина_отзыва_слов'].mean(), corpus['Длина_отзыва_слов'].mean()], color=['#A9724C', '#5B6B73'])
axes[0].set_ylabel('Средняя длина отзыва, слов')
axes[0].set_title('Длина отзыва')
axes[1].bar(['Мягкая\n(n=%d)' % len(soft), 'Корпусная\n(n=%d)' % len(corpus)],
            [soft['Эмоциональность'].mean(), corpus['Эмоциональность'].mean()], color=['#A9724C', '#5B6B73'])
axes[1].set_title('Индекс эмоциональности')
fig.suptitle('Гипотеза 4: мягкая vs корпусная мебель')
plt.tight_layout()
plt.savefig(f'{OUT}/07_furniture_type.png', dpi=130)
plt.close()

print("Готово")
