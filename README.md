Shopping Behavior Analytics: From EDA to Predictive Modeling

An end-to-end data analytics project on a retail customer shopping dataset (3,900 transactions, 18 raw attributes). It covers exploratory analysis, data cleaning and feature engineering, visualization, hypothesis testing, and a machine learning workflow that predicts customer Subscription Status.

The most important result is not a metric. The model's strong headline performance turned out to come almost entirely from two features that encode business rules, and the project shows how that was detected.

Key Results
Stage	Headline finding
EDA	Clean dataset, but 100% of subscribers are male (0 of 1,248 women subscribed), which suggests a data artifact
Cleaning	Discount Applied and Promo Code Used are 100% identical, so one was dropped; 9 features engineered
Visualization	Customer Segment (Bronze/Silver/Gold) explains spend far better than age, gender, category, season, or payment method
Statistics	7 of 8 hypothesis tests found no relationship with purchase amount; Gender × Subscription was the one strong result (Cramér's V = 0.42)
Machine Learning	Tuned Logistic Regression: 86.4% accuracy, 100% recall, 66.6% precision, F1 = 0.799, ROC-AUC = 0.908
Critical diagnostic	Without Discount Applied and Gender, performance falls to ROC-AUC ≈ 0.505 (random guessing)
Repository Structure
.
├── README.md                     <- You are here
├── PROJECT_DESCRIPTION.md        <- Detailed write-up of every stage
├── full_analysis.py              <- Statistical tests script (t-tests, chi-square, ANOVA, correlations, OLS)
├── data/
│   └── shopping_behavior_updated.csv      <- Raw dataset (add this yourself; see "Data" below)
└── reports/
    ├── Week1_EDA_Report.docx
    ├── Data_Cleaning_Report.docx
    ├── Shopping_Behavior_Analysis_Report.docx
    ├── Statistical_Analysis_Report.docx
    ├── ML_Workflow_Report.docx
    └── ML_Workflow_Capstone_Report.docx   <- Consolidated final report
Project Workflow
Exploratory Data Analysis (Week1_EDA_Report): dataset profiling, data quality checks, anomaly detection, problem definition, and analysis plan.
Data Cleaning & Pre-Processing (Data_Cleaning_Report): missing value/duplicate/outlier audit, redundancy removal, feature engineering, scaling, and one-hot encoding. Produces a 29-column cleaned file and a 46-column modeling-ready file.
Visualization (Shopping_Behavior_Analysis_Report): six charts (bar, line, scatter, heatmap, stacked bar, box plot) with business insights.
Statistical Analysis (Statistical_Analysis_Report + full_analysis.py): eight formal hypothesis tests with effect sizes and robustness checks.
Machine Learning (ML_Workflow_Report): five algorithms compared with stratified 5-fold CV, two tuned, final model evaluated on a held-out test set, plus a feature-ablation experiment.
Capstone (ML_Workflow_Capstone_Report): all stages consolidated into one narrative.

See PROJECT_DESCRIPTION.md for full detail on each stage.

Dataset
File: shopping_behavior_updated.csv
Size: 3,900 rows × 18 columns, no missing values, no duplicate Customer IDs
Fields: demographics (Age, Gender, Location), purchase details (Item, Category, Purchase Amount, Size, Color, Season), engagement (Review Rating, Subscription Status, Previous Purchases, Frequency of Purchases), and transaction metadata (Shipping Type, Discount Applied, Promo Code Used, Payment Method)
Note: the data appears to be synthetic or heavily pre-cleaned (uniform distributions, perfectly separated categories). Treat conclusions accordingly.

The CSV is not included in this repository. Download it from the original public source (commonly hosted on Kaggle as "Customer Shopping Trends") and place it in data/, or update the path in the script.

Getting Started
Requirements
Python 3.9+
pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn
bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn
Run the statistical analysis
bash
python full_analysis.py

The script reads shopping_behavior_updated.csv from the working directory, so either run it from inside data/ or edit the pd.read_csv(...) path. It prints every result and saves them to analysis_output.txt.

It runs these tests:

#	Test	Question
1	Independent t-test (+ Mann-Whitney U)	Purchase amount by gender
2	Independent t-test	Purchase amount by discount applied
3	Chi-square + Cramér's V	Gender × Subscription Status
4	Chi-square + Cramér's V	Category × Gender
5	One-way ANOVA (+ Kruskal-Wallis)	Purchase amount across categories
6	One-way ANOVA	Review rating across seasons
7	Pearson correlations	Age, previous purchases, rating vs. spend
8	Multiple linear regression (OLS)	Joint effect of six predictors on spend
Main Takeaways
Spend is essentially unpredictable from the available fields. The OLS model has R² ≈ 0.002 (p = 0.440).
Customer tier matters most. Gold, Silver, and Bronze customers spend about $78-80, $57-63, and $39-42 respectively, regardless of age.
Subscription is driven by two rule-like features. All subscribers received a discount and all are male. The ML model leans on these (Discount Applied alone accounts for about 55% of Random Forest importance).
Strong metrics can be misleading. Always run an ablation test when a few features dominate model importance.
Recommendations
Treat Discount Applied as a leakage risk and report a model without it as the realistic one.
Verify the Gender and Discount fields with the data owner before using them for targeting.
Collect richer behavioral features (engagement, tenure, service contacts) if real subscription prediction is the goal.
Organize retention strategy around customer segments rather than demographics.
Limitations
Likely synthetic data from a single retailer, so findings may not generalize
No timestamps, so no trend or recency (RFM) analysis
Single held-out test split; nested or repeated CV would give tighter estimates
No multiple-comparison correction across the eight hypothesis tests (conclusions would not change under Bonferroni)
Tech Stack

Python · pandas · NumPy · SciPy · statsmodels · scikit-learn · Matplotlib · Seaborn

