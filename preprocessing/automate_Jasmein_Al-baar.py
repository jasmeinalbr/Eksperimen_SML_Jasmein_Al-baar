# automate_Jasmein_Al-baar.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from joblib import dump

# --- Custom function for feature engineering ---
def add_features(df):
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_to_Loan"] = df["TotalIncome"] / (df["LoanAmount"] + 1)  # +1 biar aman
    return df

def preprocess_data(input_path, target_column, save_path):
    # 1. Load dataset
    df = pd.read_csv(input_path)

    # 2. Drop Loan_ID kalau ada
    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)

    # 3. Pisahkan target (Loan_Status)
    y_raw = df[target_column]
    X_raw = df.drop(columns=[target_column])

    # 4. Encode target pakai LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 5. Tambahkan fitur engineering lebih dulu
    X_fe = add_features(X_raw)

    # 6. Tentukan fitur numerik & kategorikal (setelah feature engineering)
    numeric_features = X_fe.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_fe.select_dtypes(include=["object"]).columns.tolist()

    # 7. Pipeline numerik (median + scaling)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 8. Pipeline kategorikal (mode + one-hot)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 9. ColumnTransformer gabungan
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # 10. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_fe, y, test_size=0.2, random_state=42, stratify=y
    )

    # 11. Fit-transform train, transform test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 12. Simpan pipeline
    dump(preprocessor, save_path)
    print(f"✅ Pipeline preprocessing berhasil disimpan ke: {save_path}")

    # 13. Simpan hasil preprocess ke CSV
    pd.DataFrame(X_train_processed).to_csv("preprocessing/loan_preprocessing/train_processed.csv", index=False)
    pd.DataFrame(X_test_processed).to_csv("preprocessing/loan_preprocessing/test_processed.csv", index=False)
    pd.DataFrame(y_train).to_csv("preprocessing/loan_preprocessing/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("preprocessing/loan_preprocessing/y_test.csv", index=False)
    print("✅ Hasil preprocessing berhasil disimpan ke folder loan_preprocessing/")

    return X_train_processed, X_test_processed, y_train, y_test, le

# Eksekusi fungsi jika dijalankan sebagai script utama
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = preprocess_data(
        input_path="loan_raw/loan_data.csv",
        target_column="Loan_Status",
        save_path="preprocessing/pipeline.joblib"
    )

    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
    print("Contoh label:", y_train[:10])