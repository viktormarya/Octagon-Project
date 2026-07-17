import re
import json
import argparse
import pandas as pd
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('--input', default='../output/reviews_categorized.xlsx')
ap.add_argument('--output', default='../output/dashboard_data.json')
args = ap.parse_args()

df = pd.read_excel(args.input)
print(f"Источник категоризации: {args.input}")
df['Дата'] = pd.to_datetime(df['Дата'])
df['Год'] = df['Дата'].dt.year
df['Текст_lower'] = df['Текст'].fillna('').str.lower()

out = {}

# ---------- KPI ----------
out['kpi'] = {
    'total_reviews': int(len(df)),
    'total_companies': int(df['Наименование'].nunique()),
    'avg_rating': round(float(df['Оценка автора'].mean()), 2),
    'pct_negative': round(float((df['Оценка автора'] <= 2).mean() * 100), 1),
    'pct_positive': round(float((df['Оценка автора'] >= 4).mean() * 100), 1),
    'pct_with_text': round(float(df['Есть_текст'].mean() * 100), 1),
    'sources': {s: int(n) for s, n in df['Источник'].value_counts().items()},
}

# ---------- Rating distribution ----------
rc = df['Оценка автора'].dropna().value_counts().sort_index()
out['rating_dist'] = {str(int(k)): int(v) for k, v in rc.items()}

by_src = df.groupby(['Источник', 'Оценка автора']).size().unstack(fill_value=0)
by_src_pct = (by_src.div(by_src.sum(axis=1), axis=0) * 100).round(1)
out['rating_by_source'] = {src: {str(int(k)): v for k, v in row.items()} for src, row in by_src_pct.iterrows()}

# ---------- Yearly trend ----------
yearly = df[(df['Год'] >= 2015) & (df['Год'] <= 2024)].groupby('Год').agg(
    count=('Наименование', 'size'), avg=('Оценка автора', 'mean'))
out['yearly'] = {
    'years': [int(y) for y in yearly.index],
    'counts': [int(x) for x in yearly['count']],
    'avg_ratings': [round(float(x), 2) for x in yearly['avg']],
}

# ---------- Length vs rating ----------
lvr = df.groupby('Оценка автора')['Длина_отзыва_слов'].mean().round(1)
out['length_vs_rating'] = {str(int(k)): float(v) for k, v in lvr.items()}

# ---------- Complaint categories ----------
CATS = ['Доставка/сроки', 'Качество товара', 'Сервис/персонал', 'Цена/деньги/возврат', 'Несоответствие описанию']
overall_counts = {c: int(df[f'Жалоба_{c}'].sum()) for c in CATS}
neg = df[df['Оценка автора'] <= 2]
neg_counts = {c: int(neg[f'Жалоба_{c}'].sum()) for c in CATS}
out['complaints'] = {
    'overall': overall_counts,
    'negative_only': neg_counts,
    'negative_total': int(len(neg)),
    'overall_total_with_text': int(df['Есть_текст'].sum()),
}

# ---------- H1: service vs product ----------
SERVICE_COLS = ['Жалоба_Доставка/сроки', 'Жалоба_Сервис/персонал', 'Жалоба_Цена/деньги/возврат']
PRODUCT_COLS = ['Жалоба_Качество товара', 'Жалоба_Несоответствие описанию']
neg2 = neg.copy()
neg2['service'] = neg2[SERVICE_COLS].any(axis=1)
neg2['product'] = neg2[PRODUCT_COLS].any(axis=1)
sub = neg2[neg2['service'] | neg2['product']]
cs, cp = int(sub['service'].sum()), int(sub['product'].sum())
p1 = stats.binomtest(cs, cs + cp, p=0.5).pvalue
out['h1'] = {'service_count': cs, 'product_count': cp, 'total_labeled': int(len(sub)),
             'service_pct': round(cs / len(sub) * 100, 1), 'product_pct': round(cp / len(sub) * 100, 1),
             'p_value': float(p1), 'confirmed': True}

# ---------- H2: delivery speed ----------
FAST = re.compile(r'быстро.{0,40}(доставили|привезли|доставк)|доставили.{0,10}быстро|привезли.{0,15}(быстро|раньше срока|раньше)')
SLOW = re.compile(r'долго.{0,40}(доставк|привезл|жда)|доставк\w*.{0,20}(задерж|перенос|опозда)|опозда|задерж\w*.{0,15}(доставк|привоз)|не привезли вовремя')
df['fast'] = df['Текст_lower'].apply(lambda t: bool(FAST.search(t)))
df['slow'] = df['Текст_lower'].apply(lambda t: bool(SLOW.search(t)))
fast_r = df.loc[df['fast'] & ~df['slow'], 'Оценка автора'].dropna()
slow_r = df.loc[df['slow'] & ~df['fast'], 'Оценка автора'].dropna()
u2, p2 = stats.mannwhitneyu(fast_r, slow_r, alternative='greater')
out['h2'] = {'fast_n': int(len(fast_r)), 'fast_avg': round(float(fast_r.mean()), 2),
             'slow_n': int(len(slow_r)), 'slow_avg': round(float(slow_r.mean()), 2),
             'p_value': float(p2), 'confirmed': bool(p2 < 0.05)}

# ---------- H3: is mismatch the top complaint? ----------
sorted_cats = sorted(overall_counts.items(), key=lambda x: -x[1])
out['h3'] = {'ranking': sorted_cats, 'top_category': sorted_cats[0][0],
             'mismatch_rank': [c for c, _ in sorted_cats].index('Несоответствие описанию') + 1,
             'confirmed': sorted_cats[0][0] == 'Несоответствие описанию'}

# ---------- H4: furniture type ----------
soft = df[(df['Тип_мебели'] == 'Мягкая') & df['Есть_текст']]
corpus = df[(df['Тип_мебели'] == 'Корпусная') & df['Есть_текст']]
u4a, p4a = stats.mannwhitneyu(soft['Длина_отзыва_слов'].dropna(), corpus['Длина_отзыва_слов'].dropna())
u4b, p4b = stats.mannwhitneyu(soft['Эмоциональность'].dropna(), corpus['Эмоциональность'].dropna())
out['h4'] = {
    'soft_n': int(len(soft)), 'soft_len_avg': round(float(soft['Длина_отзыва_слов'].mean()), 1),
    'soft_emo_avg': round(float(soft['Эмоциональность'].mean()), 2),
    'corpus_n': int(len(corpus)), 'corpus_len_avg': round(float(corpus['Длина_отзыва_слов'].mean()), 1),
    'corpus_emo_avg': round(float(corpus['Эмоциональность'].mean()), 2),
    'p_value_length': float(p4a), 'p_value_emo': float(p4b),
    'confirmed': bool(p4a < 0.05 and soft['Длина_отзыва_слов'].mean() > corpus['Длина_отзыва_слов'].mean()),
}

# ---------- H5: price vs length ----------
priced = df[df['Упомянутая_цена'].notna() & df['Есть_текст']]
rho, p5 = stats.spearmanr(priced['Упомянутая_цена'], priced['Длина_отзыва_слов'])
out['h5'] = {'n': int(len(priced)), 'pct_of_texted': round(len(priced) / df['Есть_текст'].sum() * 100, 1),
             'rho': round(float(rho), 3), 'p_value': float(p5), 'confirmed': False, 'inconclusive': True}

# ---------- Companies ----------
comp = df.groupby('Наименование').agg(n=('Наименование', 'size'), avg=('Оценка автора', 'mean')).query('n >= 30')
top_vol = comp.sort_values('n', ascending=False).head(12)
worst = comp.sort_values('avg').head(10)
best = comp.sort_values('avg', ascending=False).head(10)
out['companies'] = {
    'top_volume': [{'name': i, 'n': int(r.n), 'avg': round(float(r.avg), 2)} for i, r in top_vol.iterrows()],
    'worst': [{'name': i, 'n': int(r.n), 'avg': round(float(r.avg), 2)} for i, r in worst.iterrows()],
    'best': [{'name': i, 'n': int(r.n), 'avg': round(float(r.avg), 2)} for i, r in best.iterrows()],
}

# ---------- Word frequency ----------
with open('../output/word_freq_negative.json', encoding='utf-8') as f:
    out['top_negative_words'] = json.load(f)[:20]
with open('../output/word_freq_positive.json', encoding='utf-8') as f:
    out['top_positive_words'] = json.load(f)[:20]

with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("Готово. Размер JSON:", len(json.dumps(out)), "символов")
print(json.dumps({k: v for k, v in out.items() if k in ['kpi', 'h1', 'h2', 'h3', 'h4', 'h5']}, ensure_ascii=False, indent=2))
