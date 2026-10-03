import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/fake_job_postings.csv")

print("Dataset loaded successfully!")
print("Total records:", len(data))


# ==========================================
# 2. HANDLE MISSING VALUES
# ==========================================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

categorical_columns = [
    "location",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

for column in text_columns:
    data[column] = data[column].fillna("")

for column in categorical_columns:
    data[column] = data[column].fillna("Unknown")


# ==========================================
# 3. COMBINE TEXT FEATURES
# ==========================================

data["job_text"] = (
    data["title"] + " " +
    data["company_profile"] + " " +
    data["description"] + " " +
    data["requirements"] + " " +
    data["benefits"]
)


# ==========================================
# 4. SELECT FEATURES
# ==========================================

X = data[
    ["job_text"] + categorical_columns
]

y = data["fraudulent"]


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. FEATURE PROCESSING
# ==========================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "text",
            TfidfVectorizer(
                stop_words="english",
                max_features=50000
            ),
            "job_text"
        ),

        (
            "category",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )

    ]

)


# ==========================================
# 7. MACHINE LEARNING MODEL
# ==========================================

model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000
            )
        )

    ]

)


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nTraining improved model...")

model.fit(X_train, y_train)


# ==========================================
# 9. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 10. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print("MODEL TRAINING COMPLETED")
print("================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Real Job",
            "Fake Job"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "model/fake_job_model_v2.pkl"
)

print("\nImproved model saved successfully!")
