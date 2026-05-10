import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
from openai import OpenAI
import json

# ── PAGE CONFIG ──────────────────────────────────────────────────
st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗", layout="centered")

# ── LOAD API KEY FROM SECRETS ────────────────────────────────────
api_key = st.secrets["OPENAI_API_KEY"]

# ── LOAD MODELS ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('model_xgb.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('model_lr.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return xgb, lr, encoders, explainer, feature_names

xgb, lr, encoders, explainer, feature_names = load_models()

# ── CONSTANTS ────────────────────────────────────────────────────
feature_name_map = {
    'age_x_odometer': 'Relationship between vehicle age and mileage',
    'year':           'Model year',
    'age':            'Vehicle age',
    'odometer':       'Mileage',
    'drive':          'Drive type (FWD/RWD/AWD/4WD)',
    'type':           'Vehicle type (truck/sedan/SUV etc.)',
    'manufacturer':   'Brand/manufacturer',
    'fuel':           'Fuel type',
    'model':          'Specific model',
    'state':          'State/region',
    'paint_color':    'Exterior color',
    'condition':      'Vehicle condition',
    'transmission':   'Transmission type'
}

condition_order = {'new':5, 'like new':4, 'excellent':3,
                   'good':2, 'fair':1, 'salvage':0, 'missing':-1}

training_medians = {'year': 2013, 'odometer': 94000}

# ── AI FUNCTIONS ─────────────────────────────────────────────────
def extract_features_from_text(user_description):
    client = OpenAI(api_key=api_key)
    prompt = f"""
    Extract car features from this description and return ONLY a JSON object with exactly these keys:
    year, manufacturer, model, condition, odometer, fuel, transmission, drive, type, paint_color, state

    Rules:
    - year: integer (e.g. 2015)
    - manufacturer: lowercase brand name (e.g. "honda", "ford", "toyota")
    - model: lowercase model name (e.g. "civic", "f-150")
    - condition: one of: "new", "like new", "excellent", "good", "fair", "salvage"
    - odometer: integer miles (e.g. 80000)
    - fuel: one of: "gas", "diesel", "hybrid", "electric", "other"
    - transmission: one of: "automatic", "manual", "other"
    - drive: one of: "fwd", "rwd", "4wd", "awd"
    - type: one of: "sedan", "suv", "truck", "pickup", "coupe", "hatchback", "wagon", "van", "convertible", "other"
    - paint_color: lowercase color (e.g. "black", "white", "silver", "blue")
    - state: lowercase 2-letter state code (e.g. "ca", "ny", "tx") or "missing" if unknown

    If any field is unknown, use "missing" for strings or -1 for numbers.
    Description: "{user_description}"
    Return ONLY the JSON, no explanation, no markdown, no backticks.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    text = response.choices[0].message.content.strip()
    text = text.replace('```json', '').replace('```', '').strip()
    return json.loads(text)

def generate_explanation(user_description, predicted_price, features, top3):
    client = OpenAI(api_key=api_key)
    prompt = f"""
    A user described their car as: "{user_description}"
    Our ML model predicted the fair market value at ${predicted_price:,.0f}.
    Key features: Year: {features.get('year')}, Manufacturer: {features.get('manufacturer')},
    Model: {features.get('model')}, Condition: {features.get('condition')}, Odometer: {features.get('odometer')} miles
    Top factors that influenced the price: {top3}
    Write a 2-3 sentence plain-English explanation of why this car is priced at ${predicted_price:,.0f}.
    Be specific, mention the car details, and reference the top factors.
    Do not use bullet points. Write in a friendly, helpful tone.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def predict_price(features):
    year = features.get('year') if features.get('year') not in [None, -1] else training_medians['year']
    odometer = features.get('odometer') if features.get('odometer') not in [None, -1] else training_medians['odometer']
    age = 2024 - year
    age_x_odometer = age * odometer

    row = {
        'year': year,
        'manufacturer': features.get('manufacturer', 'missing'),
        'model': features.get('model', 'missing'),
        'condition': condition_order.get(features.get('condition', 'missing'), -1),
        'odometer': odometer,
        'fuel': features.get('fuel', 'gas'),
        'transmission': features.get('transmission', 'automatic'),
        'drive': features.get('drive', 'missing'),
        'type': features.get('type', 'missing'),
        'paint_color': features.get('paint_color', 'missing'),
        'state': features.get('state', 'missing'),
        'age': age,
        'age_x_odometer': age_x_odometer
    }

    for col in ['manufacturer', 'model', 'fuel', 'transmission',
                'drive', 'type', 'paint_color', 'state']:
        le = encoders[col]
        val = row[col]
        if val in le.classes_:
            row[col] = le.transform([val])[0]
        else:
            row[col] = le.transform(['missing'])[0] if 'missing' in le.classes_ else 0

    input_df = pd.DataFrame([row])[feature_names]
    predicted_price = xgb.predict(input_df)[0]

    shap_vals = explainer.shap_values(input_df)
    shap_series = pd.Series(np.abs(shap_vals[0]), index=feature_names)
    top3_raw = shap_series.nlargest(3).index.tolist()
    top3_readable = [feature_name_map.get(f, f) for f in top3_raw]

    return predicted_price, top3_readable

# ── UI ───────────────────────────────────────────────────────────
st.title("🚗 Used Car Price Estimator")
st.markdown("Describe your car in plain English and get an instant fair market value estimate.")

st.markdown("### Describe your car")
user_input = st.text_area(
    "Be as detailed as you like — year, make, model, mileage, condition, color, state, etc.",
    placeholder="e.g. 2015 Honda Civic, 80k miles, good condition, automatic, blue, California",
    height=100
)

if st.button("Estimate Price 🚗"):
    if not user_input:
        st.error("Please describe your car first.")
    else:
        with st.spinner("Analyzing your car..."):
            try:
                features = extract_features_from_text(user_input)
                predicted_price, top3 = predict_price(features)
                explanation = generate_explanation(user_input, predicted_price, features, top3)

                st.success("Done!")

                st.markdown("## 💰 Estimated Fair Market Value")
                st.markdown(f"# ${predicted_price:,.0f}")

                st.markdown("### 🔍 Top Factors Driving This Price")
                for i, factor in enumerate(top3, 1):
                    st.markdown(f"**{i}.** {factor}")

                st.markdown("### 📝 Explanation")
                st.info(explanation)

                st.markdown("### 📋 Extracted Car Details")
                display_features = {k: v for k, v in features.items() if v != 'missing' and v != -1}
                st.json(display_features)

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
