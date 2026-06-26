import pandas as pd

def preprocess_input(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):

    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 0 if self_employed == "No" else 1

    input_df = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [education_encoded],
        'self_employed': [self_employed_encoded],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    })

    return input_df