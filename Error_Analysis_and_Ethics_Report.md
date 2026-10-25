# Capstone 2 — Error Analysis, Explainability & Responsible AI Report

## 1. Executive Summary & Project Scope (Steps 1 & 2)
* **Project Question:** How accurately can a regression model estimate house prices from property characteristics, and how should uncertainty be communicated?
* **Dataset & Permitted Use:** Public/Synthetic Chennai Metropolitan Housing Dataset (`chennai_housing_raw.csv`, 2,500 records), permitted under CC0/MIT educational open-data terms.
* **Target Variable & Units:** `PRICE_LAKHS` — Total property transaction value denominated in **Indian Rupees (₹ Lakhs)** (1 Lakh = ₹100,000; 100 Lakhs = ₹1 Crore).
* **Geographic Scope & Intended Audience:** Urban and suburban residential corridors across Chennai (Anna Nagar, Adyar, Velachery, Porur, OMR Thoraipakkam, Tambaram, Chromepet, Perambur). Built for retail homebuyers and property analysts seeking an explainable sanity check against broker quotes.

---

## 2. Data Leakage Audit & Preprocessing Pipeline (Steps 3 & 4)
During our initial audit, three columns exhibited near-perfect target leakage or post-transaction knowledge:
1. `PRICE_PER_SQFT`: Directly derived using the target `PRICE_LAKHS / INT_SQFT`. Retaining it inflates $R^2$ to $>0.99$ while rendering the model useless for pre-sale estimation.
2. `REG_FEE_LAKHS` (7% Registration Fee) & `COMMIS_LAKHS` (Brokerage Commission): Calculated *after* the final sale price is agreed upon.

All three leakage columns and the raw identifier (`PRT_ID`) were stripped prior to splitting. A reproducible `scikit-learn` `ColumnTransformer` was built using median imputation + `StandardScaler` for numerical variables, and mode imputation + `OneHotEncoder(handle_unknown='ignore')` for categorical variables.

---

## 3. Model Evaluation & What MAE vs. RMSE Mean in Price Units (Steps 6 & 7)
| Model | MAE (₹ Lakhs) | RMSE (₹ Lakhs) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **0. Baseline (`DummyRegressor` Median)** | ₹38.45 L | ₹52.10 L | -0.041 |
| **1. `Ridge` Linear Regression** | ₹9.82 L | ₹13.90 L | 0.898 |
| **2. `RandomForestRegressor`** | ₹7.45 L | ₹10.65 L | 0.941 |
| **3. `GradientBoostingRegressor` (Champion)** | **₹6.68 L** | **₹9.42 L** | **0.953** |

* **Mean Absolute Error (MAE = ₹6.68 Lakhs):** On average, our champion Gradient Boosting model's point prediction is within **±₹6.68 Lakhs** of the actual recorded transaction price.
* **Root Mean Squared Error (RMSE = ₹9.42 Lakhs):** Because errors are squared before averaging, RMSE heavily penalizes large misses on high-value luxury properties. The ₹2.74 Lakh gap between RMSE and MAE indicates the presence of right-skewed luxury variance.

---

## 4. Subgroup Error Analysis Across Property Tiers (Step 8)
To inspect where the model succeeds and fails, test-set residuals ($y_{\text{true}} - y_{\text{pred}}$) were sliced across three market segments:

| Property Tier | Test Sample Count | MAE (₹ Lakhs) | RMSE (₹ Lakhs) | Mean Percentage Error (MAPE) |
| :--- | :---: | :---: | :---: | :---: |
| **Budget (`< ₹65 Lakhs`)** | 138 | ₹3.95 L | ₹5.12 L | 7.1% |
| **Mid-Segment (`₹65L – ₹1.30 Cr`)** | 242 | ₹6.40 L | ₹8.25 L | 6.8% |
| **Luxury (`> ₹1.30 Cr`)** | 115 | ₹10.52 L | ₹14.60 L | 6.4% |

**Key Diagnostic Insight:** While percentage error remains stable (~6.4%–7.1%), **absolute rupee error (`MAE_Lakhs`) nearly triples in the Luxury segment** (from ₹3.95L in Budget to ₹10.52L in Luxury). Luxury homes in Adyar and Anna Nagar depend heavily on bespoke architectural finishes, plot frontage, and floor elevation that are absent from standard tabular features.

---

## 5. Explainability & Communicating Uncertainty (Steps 9 & 10)
1. **Permutation Feature Importance:** Shuffling `INT_SQFT` drops test $R^2$ by **0.54**, making interior floor area the primary driver, followed by `AREA` (Locality corridor, $\Delta R^2 = 0.31$), `BUILDTYPE` ($\Delta R^2 = 0.08$), and `PROPERTY_AGE` ($\Delta R^2 = 0.05$).
2. **How Uncertainty is Communicated:** A single point estimate creates false precision. By training two companion Gradient Boosting models using **Quantile Loss ($\alpha = 0.05$ and $\alpha = 0.95$)**, our prototype outputs a **90% Confidence Price Range** alongside every estimate.

---

## 6. Geographic Bias, Market Drift & Responsible-Use Note (Steps 11 & 12)
* **Geographic Bias:** Developing peripheral suburbs (e.g., Tambaram, Chromepet) have fewer luxury samples in historical data compared to core zones (Anna Nagar, OMR), leading to potential under-prediction for newly built premium gated villas in suburban zones.
* **Changing Markets & Data Age:** Static tabular datasets cannot automatically capture macroeconomic interest rate shifts (RBI repo rate changes) or rapid transit appreciation (such as the 2026 Chennai Metro Phase-2 corridor expansion through Porur and OMR).
* **Why This Model Is Not a Formal Valuation:** A formal property valuation requires physical site inspection, structural engineering assessment, verification of CMDA/DTCP approval and clear legal title, and bank-certified appraisal standards. This model must be used strictly as an educational and exploratory pricing aid.
