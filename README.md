# PropPulse Chennai — House Price Estimation & Uncertainty Engine
**AI & ML Capstone 2 — House Price Estimation Using Regression**

An end-to-end machine learning pipeline and interactive web prototype that estimates residential property prices across Chennai (in ₹ Lakhs) and communicates prediction uncertainty using **90% Quantile Regression Intervals**.

---

## 📂 Repository Structure
```text
├── data.py                              # Step 1 & 3: Dataset generator with intentional leakage columns
├── pipeline.py                          # Steps 2–10: EDA, Preprocessing Pipeline, 4 Models, Quantile Uncertainty & XAI
├── app.py                               # Step 9: Interactive Streamlit Price & Uncertainty Prototype
├── Error_Analysis_and_Ethics_Report.md  # Steps 7, 8, 11, 12: Subgroup error slicing & Responsible AI report
├── requirements.txt                     # Reproducible Python package dependencies
└── README.md                            # Setup & Reproducibility Guide
