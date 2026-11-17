import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("shopping_behavior_updated.csv")
out = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)

log("="*70)
log("TEST 1: Independent t-test - Purchase Amount by Gender")
log("="*70)
m = df[df.Gender=='Male']['Purchase Amount (USD)']
f = df[df.Gender=='Female']['Purchase Amount (USD)']
t, p = stats.ttest_ind(m, f, equal_var=True)
pooled_sd = np.sqrt(((len(m)-1)*m.std()**2 + (len(f)-1)*f.std()**2)/(len(m)+len(f)-2))
d = (m.mean()-f.mean())/pooled_sd
diff = m.mean()-f.mean()
se = pooled_sd*np.sqrt(1/len(m)+1/len(f))
ci = (diff - 1.96*se, diff+1.96*se)
log(f"Male mean={m.mean():.2f} (n={len(m)}), Female mean={f.mean():.2f} (n={len(f)})")
log(f"t={t:.4f}, p={p:.4f}, Cohen's d={d:.4f}, 95% CI of diff={ci}")
u, pu = stats.mannwhitneyu(m, f)
log(f"[Robustness] Mann-Whitney U={u:.1f}, p={pu:.4f}")

log("")
log("="*70)
log("TEST 2: Independent t-test - Purchase Amount by Discount Applied")
log("="*70)
y = df[df['Discount Applied']=='Yes']['Purchase Amount (USD)']
n = df[df['Discount Applied']=='No']['Purchase Amount (USD)']
t, p = stats.ttest_ind(y, n, equal_var=True)
pooled_sd = np.sqrt(((len(y)-1)*y.std()**2 + (len(n)-1)*n.std()**2)/(len(y)+len(n)-2))
d = (y.mean()-n.mean())/pooled_sd
diff = y.mean()-n.mean()
se = pooled_sd*np.sqrt(1/len(y)+1/len(n))
ci = (diff - 1.96*se, diff+1.96*se)
log(f"Discount Yes mean={y.mean():.2f} (n={len(y)}), Discount No mean={n.mean():.2f} (n={len(n)})")
log(f"t={t:.4f}, p={p:.4f}, Cohen's d={d:.4f}, 95% CI of diff={ci}")

log("")
log("="*70)
log("TEST 3: Chi-square - Gender x Subscription Status")
log("="*70)
ct = pd.crosstab(df['Gender'], df['Subscription Status'])
chi2, p, dof, exp = stats.chi2_contingency(ct)
n = ct.sum().sum()
cramers_v = np.sqrt(chi2/(n*(min(ct.shape)-1)))
log(ct.to_string())
log(f"chi2={chi2:.4f}, dof={dof}, p={p:.4e}, Cramer's V={cramers_v:.4f}")
log(f"Expected counts:\n{np.round(exp,2)}")

log("")
log("="*70)
log("TEST 4: Chi-square - Category x Gender")
log("="*70)
ct2 = pd.crosstab(df['Category'], df['Gender'])
chi2, p, dof, exp = stats.chi2_contingency(ct2)
n = ct2.sum().sum()
cramers_v = np.sqrt(chi2/(n*(min(ct2.shape)-1)))
log(ct2.to_string())
log(f"chi2={chi2:.4f}, dof={dof}, p={p:.4f}, Cramer's V={cramers_v:.4f}")

log("")
log("="*70)
log("TEST 5: One-way ANOVA - Purchase Amount across Category")
log("="*70)
groups = [g['Purchase Amount (USD)'].values for _, g in df.groupby('Category')]
fstat, p = stats.f_oneway(*groups)
grand_mean = df['Purchase Amount (USD)'].mean()
ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in groups)
ss_total = sum((df['Purchase Amount (USD)']-grand_mean)**2)
eta_sq = ss_between/ss_total
log(df.groupby('Category')['Purchase Amount (USD)'].agg(['mean','std','count']).to_string())
log(f"F={fstat:.4f}, p={p:.4f}, eta-squared={eta_sq:.5f}")
h, ph = stats.kruskal(*groups)
log(f"[Robustness] Kruskal-Wallis H={h:.4f}, p={ph:.4f}")

log("")
log("="*70)
log("TEST 6: One-way ANOVA - Review Rating across Season")
log("="*70)
groups2 = [g['Review Rating'].values for _, g in df.groupby('Season')]
fstat, p = stats.f_oneway(*groups2)
grand_mean2 = df['Review Rating'].mean()
ss_between2 = sum(len(g)*(g.mean()-grand_mean2)**2 for g in groups2)
ss_total2 = sum((df['Review Rating']-grand_mean2)**2)
eta_sq2 = ss_between2/ss_total2
log(df.groupby('Season')['Review Rating'].agg(['mean','std','count']).to_string())
log(f"F={fstat:.4f}, p={p:.4f}, eta-squared={eta_sq2:.5f}")

log("")
log("="*70)
log("TEST 7: Pearson correlations")
log("="*70)
for col in ['Age','Previous Purchases','Review Rating']:
    r, p = stats.pearsonr(df[col], df['Purchase Amount (USD)'])
    log(f"Purchase Amount vs {col}: r={r:.4f}, p={p:.4f}")
r, p = stats.pearsonr(df['Age'], df['Review Rating'])
log(f"Age vs Review Rating: r={r:.4f}, p={p:.4f}")
r, p = stats.pearsonr(df['Age'], df['Previous Purchases'])
log(f"Age vs Previous Purchases: r={r:.4f}, p={p:.4f}")

log("")
log("="*70)
log("TEST 8: Multiple Linear Regression - Purchase Amount ~ predictors")
log("="*70)
model_df = df.copy()
model_df['Gender_Male'] = (model_df['Gender']=='Male').astype(int)
model_df['Discount_Yes'] = (model_df['Discount Applied']=='Yes').astype(int)
model_df['Subscribed_Yes'] = (model_df['Subscription Status']=='Yes').astype(int)
formula = 'Q("Purchase Amount (USD)") ~ Age + Q("Previous Purchases") + Q("Review Rating") + Gender_Male + Discount_Yes + Subscribed_Yes'
model = smf.ols(formula, data=model_df).fit()
log(model.summary().as_text())

with open("analysis_output.txt","w") as f:
    f.write("\n".join(out))
