# File 1: generate_chennai_data.py
import numpy as np
import pandas as pd

def generate_chennai_housing_dataset(n_samples=2500, seed=42):
    np.random.seed(seed)

    # Chennai Localities with realistic 2026 base rates (INR per SqFt)
    localities = {
        "Anna Nagar": 11800,
        "Adyar": 13200,
        "Velachery": 8400,
        "Porur": 7100,
        "OMR Thoraipakkam": 7800,
        "Tambaram": 5200,
        "Chromepet": 6000,
        "Perambur": 6700,
    }
    area_names = list(localities.keys())
    area_probs = [0.12, 0.08, 0.16, 0.18, 0.18, 0.12, 0.08, 0.08]

    chosen_areas = np.random.choice(area_names, size=n_samples, p=area_probs)
    base_rates = np.array([localities[a] for a in chosen_areas])

    # Property physical characteristics
    int_sqft = np.random.normal(loc=1250, scale=420, size=n_samples).clip(480, 4200).round(0)
    
    # Bedrooms correlated with SqFt
    n_bedroom = np.where(int_sqft < 850, 1,
                np.where(int_sqft < 1450, 2,
                np.where(int_sqft < 2200, 3, 4))).astype(float)
    
    n_bathroom = np.clip(n_bedroom - np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]), 1, 4).astype(float)
    n_room = n_bedroom + n_bathroom + np.random.choice([1, 2], size=n_samples)

    property_age = np.random.exponential(scale=8.5, size=n_samples).clip(0, 35).round(1)
    dist_mainroad = np.random.uniform(10, 500, size=n_samples).round(0)

    buildtype = np.random.choice(
        ["Residential", "Commercial", "Mixed"],
        size=n_samples,
        p=[0.74, 0.16, 0.10]
    )
    park_facil = np.random.choice(["Yes", "No"], size=n_samples, p=[0.78, 0.22])
    sale_cond = np.random.choice(
        ["Normal Sale", "Family", "AbNormal", "Partial", "AdjLand"],
        size=n_samples,
        p=[0.65, 0.10, 0.10, 0.08, 0.07]
    )

    # Calculate True Underlying Price (in INR Lakhs)
    build_multiplier = np.where(buildtype == "Commercial", 1.28, np.where(buildtype == "Mixed", 1.12, 1.0))
    park_bonus = np.where(park_facil == "Yes", 4.5, 0.0)
    age_depreciation = np.maximum(0.68, 1.0 - (property_age * 0.011))
    road_discount = np.where(dist_mainroad > 300, -2.5, 1.5)

    # Base calculation in Lakhs + heteroscedastic market noise (luxury properties vary more)
    raw_price_lakhs = (
        ((int_sqft * base_rates) / 100000.0) * build_multiplier * age_depreciation
        + (n_bedroom * 3.2)
        + park_bonus
        + road_discount
    )
    market_noise = np.random.normal(0, raw_price_lakhs * 0.085, size=n_samples)
    price_lakhs = np.clip(raw_price_lakhs + market_noise, 18.0, None).round(2)

    # Inject 1.5% Extreme Luxury / Data Entry Outliers (Step 3 requirement)
    outlier_idx = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    price_lakhs[outlier_idx] = price_lakhs[outlier_idx] * np.random.uniform(2.2, 3.5, size=len(outlier_idx))
    price_lakhs = price_lakhs.round(2)

    # Create Intentional Target Leakage Features (Must be audited & dropped in Step 3!)
    price_per_sqft = ((price_lakhs * 100000) / int_sqft).round(0)
    reg_fee_lakhs = (price_lakhs * 0.07 + np.random.normal(0, 0.15, size=n_samples)).clip(0.5, None).round(2)
    commis_lakhs = (price_lakhs * 0.015 + np.random.normal(0, 0.08, size=n_samples)).clip(0.1, None).round(2)

    df = pd.DataFrame({
        "PRT_ID": [f"CHN_{10000 + i}" for i in range(n_samples)],
        "AREA": chosen_areas,
        "INT_SQFT": int_sqft,
        "N_BEDROOM": n_bedroom,
        "N_BATHROOM": n_bathroom,
        "N_ROOM": n_room,
        "PROPERTY_AGE": property_age,
        "DIST_MAINROAD": dist_mainroad,
        "BUILDTYPE": buildtype,
        "PARK_FACIL": park_facil,
        "SALE_COND": sale_cond,
        # Leakage columns
        "PRICE_PER_SQFT": price_per_sqft,
        "REG_FEE_LAKHS": reg_fee_lakhs,
        "COMMIS_LAKHS": commis_lakhs,
        # Target variable
        "PRICE_LAKHS": price_lakhs
    })

    # Inject Realistic Missing Values (~3-5% in selected columns for Step 3 & 4)
    for col, frac in [("N_BEDROOM", 0.03), ("N_BATHROOM", 0.04), ("PARK_FACIL", 0.035), ("PROPERTY_AGE", 0.025)]:
        mask = np.random.choice(n_samples, size=int(n_samples * frac), replace=False)
        df.loc[mask, col] = np.nan

    df.to_csv("chennai_housing_raw.csv", index=False)
    print(f"[SUCCESS] Generated 'chennai_housing_raw.csv' with shape {df.shape}")
    return df

if __name__ == "__main__":
    generate_chennai_housing_dataset()
