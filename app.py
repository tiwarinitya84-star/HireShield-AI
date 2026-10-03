from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model/fake_job_model_v2.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/performance")
def performance():
    return render_template("performance.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------
    # Get form data
    # -----------------------------

    title = request.form.get("title", "")
    location = request.form.get("location", "")
    employment_type = request.form.get("employment_type", "")
    required_education = request.form.get("required_education", "")
    required_experience = request.form.get("required_experience", "")
    industry = request.form.get("industry", "")
    function = request.form.get("function", "")

    company_profile = request.form.get("company_profile", "")
    description = request.form.get("description", "")
    requirements = request.form.get("requirements", "")
    benefits = request.form.get("benefits", "")


    # -----------------------------
    # Combine text fields
    # -----------------------------

    job_text = (
        title + " " +
        company_profile + " " +
        description + " " +
        requirements + " " +
        benefits
    )


    # -----------------------------
    # Prepare input for ML model
    # -----------------------------

    input_data = pd.DataFrame([{
        "job_text": job_text,
        "location": location,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function
    }])


    # -----------------------------
    # Make prediction
    # -----------------------------

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]


    # -----------------------------
    # Result and confidence
    # -----------------------------

    if prediction == 1:

        result = "FAKE JOB"
        confidence = probability[1] * 100

        risk_level = "HIGH"

        risk_message = (
            "The AI model detected patterns that may indicate "
            "a potentially fraudulent job posting."
        )

    else:

        result = "REAL JOB"
        confidence = probability[0] * 100

        risk_level = "LOW"

        risk_message = (
            "The AI model did not detect strong indicators "
            "of a fraudulent job posting."
        )


    # -----------------------------
    # Risk indicators
    # -----------------------------

    risk_indicators = []

    full_text = (
        title + " " +
        company_profile + " " +
        description + " " +
        requirements + " " +
        benefits
    ).lower()


    if "no experience" in full_text:
        risk_indicators.append("No experience required")


    if "work from home" in full_text:
        risk_indicators.append("Work-from-home claim")


    if (
        "earn $" in full_text
        or "high salary" in full_text
        or "high income" in full_text
    ):
        risk_indicators.append(
            "Unusually high earning/salary claim"
        )


    if (
        "start immediately" in full_text
        or "immediate joining" in full_text
    ):
        risk_indicators.append(
            "Immediate joining language"
        )


    if (
        "payment every day" in full_text
        or "daily payment" in full_text
    ):
        risk_indicators.append(
            "Daily payment claim"
        )


    # -----------------------------
    # Safety checklist
    # -----------------------------

    safety_checks = []


    if not company_profile.strip():

        safety_checks.append({
            "title": "Company information is missing",
            "status": "warning"
        })

    else:

        safety_checks.append({
            "title": "Company information provided",
            "status": "good"
        })


    if not location.strip():

        safety_checks.append({
            "title": "Job location is not specified",
            "status": "warning"
        })

    else:

        safety_checks.append({
            "title": "Job location provided",
            "status": "good"
        })


    if not employment_type.strip():

        safety_checks.append({
            "title": "Employment type is not specified",
            "status": "warning"
        })

    else:

        safety_checks.append({
            "title": "Employment type provided",
            "status": "good"
        })


    if not requirements.strip():

        safety_checks.append({
            "title": "Job requirements are missing",
            "status": "warning"
        })

    else:

        safety_checks.append({
            "title": "Job requirements provided",
            "status": "good"
        })


    if not description.strip():

        safety_checks.append({
            "title": "Job description is missing",
            "status": "warning"
        })

    else:

        safety_checks.append({
            "title": "Job description provided",
            "status": "good"
        })


    # -----------------------------
    # Send result to result.html
    # -----------------------------

    return render_template(
        "result.html",
        result=result,
        confidence=round(confidence, 2),
        risk_level=risk_level,
        risk_message=risk_message,
        safety_checks=safety_checks,
        risk_indicators=risk_indicators
    )


# -----------------------------
# Run Flask application
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)