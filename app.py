from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
# Auto-train karo agar pkl files nahi hain
# (Render.com deploy ke liye)
# ─────────────────────────────────────────────
if not os.path.exists('crop_model.pkl'):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('Crop_recommendation.csv')

    def generate_samples(n, N, P, K, temp, humidity, ph, rainfall, label):
        return pd.DataFrame({
            'N':           np.random.randint(N-10, N+10, n),
            'P':           np.random.randint(P-10, P+10, n),
            'K':           np.random.randint(K-10, K+10, n),
            'temperature': np.random.uniform(temp-2, temp+2, n),
            'humidity':    np.random.uniform(humidity-5, humidity+5, n),
            'ph':          np.random.uniform(ph-0.3, ph+0.3, n),
            'rainfall':    np.random.uniform(rainfall-15, rainfall+15, n),
            'label':       label
        })

    new_crops = pd.concat([
        generate_samples(100, 60, 40, 20, 15.0, 55.0, 6.5, 70.0,  'wheat'),
        generate_samples(100, 40, 25, 15, 28.0, 50.0, 7.2, 55.0,  'bajra'),
        generate_samples(100, 25, 50, 30, 27.0, 65.0, 6.2, 95.0,  'groundnut'),
        generate_samples(100, 35, 60, 40, 25.0, 68.0, 6.8, 110.0, 'soybean'),
        generate_samples(100, 80, 40, 50, 30.0, 80.0, 6.5, 180.0, 'sugarcane'),
        generate_samples(100, 30, 20, 15, 27.0, 48.0, 7.0, 50.0,  'jowar'),
        generate_samples(100, 50, 30, 10, 18.0, 58.0, 6.8, 60.0,  'mustard'),
    ])

    df = pd.concat([df, new_crops], ignore_index=True)

    X = df.drop('label', axis=1)
    y = df['label']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_encoded)

    pickle.dump(rf, open('crop_model.pkl', 'wb'))
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    pickle.dump(le, open('label_encoder.pkl', 'wb'))

    print("Model trained and saved successfully!")

# ─────────────────────────────────────────────
# Load saved models
# ─────────────────────────────────────────────
model   = pickle.load(open('crop_model.pkl', 'rb'))
scaler  = pickle.load(open('scaler.pkl', 'rb'))
le      = pickle.load(open('label_encoder.pkl', 'rb'))

FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ─────────────────────────────────────────────
# Crop one-line descriptions (29 crops)
# ─────────────────────────────────────────────
CROP_INFO = {
    # Original 22 crops
    'rice':        'Best suited for waterlogged areas with high humidity and heavy rainfall.',
    'maize':       'Thrives in warm climate with moderate rainfall and well-drained soil.',
    'chickpea':    'Ideal for dry and cool conditions with very low water requirement.',
    'kidneybeans': 'Grows well in moist fertile soil with moderate temperature.',
    'pigeonpeas':  'Drought resistant crop suitable for semi-arid tropical regions.',
    'mothbeans':   'Best for arid and semi-arid regions with sandy and dry soil.',
    'mungbean':    'Suitable for tropical climate with short growing season.',
    'blackgram':   'Thrives in humid conditions with moderate to high rainfall.',
    'lentil':      'Prefers cool climate with well-drained loamy and fertile soil.',
    'pomegranate': 'Best for dry climate with hot summers and mild winters.',
    'banana':      'Requires tropical climate with high humidity and consistent rainfall.',
    'mango':       'Thrives in tropical and subtropical regions with dry winters.',
    'grapes':      'Best suited for hot dry summers and cool dry winters.',
    'watermelon':  'Requires warm temperature with sandy well-drained light soil.',
    'muskmelon':   'Grows best in warm dry climate with low humidity levels.',
    'apple':       'Requires cold hilly climate with well-drained deep fertile soil.',
    'orange':      'Best for subtropical climate with moderate and well-distributed rainfall.',
    'papaya':      'Thrives in tropical climate and cannot tolerate waterlogging at all.',
    'coconut':     'Ideal for coastal tropical areas with high humidity and rainfall.',
    'cotton':      'Requires warm climate with deep black soil and moderate rainfall.',
    'jute':        'Grows best in hot humid climate with very heavy seasonal rainfall.',
    'coffee':      'Ideal for tropical highlands with moderate temperature and shade.',
    # New 7 crops
    'wheat':       'Best for cool dry climate with well-drained loamy fertile soil.',
    'bajra':       'Highly drought tolerant — grows well in hot arid sandy soil.',
    'groundnut':   'Thrives in warm climate with well-drained sandy loam soil.',
    'soybean':     'Grows well in warm humid climate with well-drained fertile soil.',
    'sugarcane':   'Requires tropical climate with high rainfall and deep fertile soil.',
    'jowar':       'Drought tolerant crop ideal for hot dry semi-arid conditions.',
    'mustard':     'Best suited for cool dry climate with well-drained loamy soil.',
}

# ─────────────────────────────────────────────
# Route 1: Crop Recommendation
# ─────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # N, P, K — farmer nahi jaanta, dataset average use karo
    features = np.array([[
        data.get('N', 52),           # default: dataset average
        data.get('P', 53),           # default: dataset average
        data.get('K', 48),           # default: dataset average
        data['temperature'],
        data['humidity'],
        data['ph'],
        data['rainfall']
    ]])

    scaled    = scaler.transform(features)
    pred      = model.predict(scaled)
    prob      = model.predict_proba(scaled)
    crop_name = le.inverse_transform(pred)[0]
    confidence = round(prob.max() * 100, 2)

    # Top 3 crops
    top3_idx = prob[0].argsort()[-3:][::-1]
    top3 = [
        {
            "crop":       le.classes_[i],
            "confidence": round(prob[0][i] * 100, 2)
        }
        for i in top3_idx
    ]

    # Key factor — dominant feature
    top_feature = FEATURE_NAMES[int(np.argmax(model.feature_importances_))]

    # Low confidence warning
    warning = None
    if confidence < 50:
        warning = "Low confidence prediction — please verify your input values or consult an agronomist."

    return jsonify({
        'recommended_crop': crop_name,
        'confidence':       confidence,
        'description':      CROP_INFO.get(crop_name, 'No description available.'),
        'key_factor':       top_feature,
        'top3':             top3,
        'warning':          warning
    })


# ─────────────────────────────────────────────
# Route 2: Fertilizer Recommendation
# ─────────────────────────────────────────────
@app.route('/fertilizer', methods=['POST'])
def fertilizer():
    data = request.json
    N    = data.get('N', 0)
    P    = data.get('P', 0)
    K    = data.get('K', 0)
    crop = data.get('crop', 'Unknown')

    # NPK level determine karo
    def level(val):
        if val < 30:   return 'low'
        elif val > 70: return 'high'
        else:          return 'medium'

    n_lvl = level(N)
    p_lvl = level(P)
    k_lvl = level(K)

    # Fertilizer database
    FERTILIZER_DB = {
        ('low',    'low',    'low'):    {
            "fertilizer":          "NPK 19:19:19",
            "npk_ratio":           "19:19:19",
            "application_method":  "Broadcast and mix into soil before sowing",
            "best_season":         "Kharif and Rabi both"
        },
        ('low',    'low',    'medium'): {
            "fertilizer":          "DAP + Urea",
            "npk_ratio":           "18:46:0 + 46:0:0",
            "application_method":  "Basal application at sowing time",
            "best_season":         "Kharif (June-July)"
        },
        ('low',    'low',    'high'):   {
            "fertilizer":          "DAP + Urea",
            "npk_ratio":           "18:46:0 + 46:0:0",
            "application_method":  "Split application — half at sowing, half at tillering",
            "best_season":         "Rabi (October-November)"
        },
        ('low',    'medium', 'low'):    {
            "fertilizer":          "Urea + MOP",
            "npk_ratio":           "46:0:0 + 0:0:60",
            "application_method":  "Top dressing after first irrigation",
            "best_season":         "Kharif (June-July)"
        },
        ('low',    'medium', 'medium'): {
            "fertilizer":          "Urea",
            "npk_ratio":           "46:0:0",
            "application_method":  "Top dressing in 2 splits",
            "best_season":         "Kharif and Rabi both"
        },
        ('low',    'medium', 'high'):   {
            "fertilizer":          "Urea",
            "npk_ratio":           "46:0:0",
            "application_method":  "Foliar spray or soil application",
            "best_season":         "Rabi (October-November)"
        },
        ('low',    'high',   'low'):    {
            "fertilizer":          "Urea + MOP",
            "npk_ratio":           "46:0:0 + 0:0:60",
            "application_method":  "Broadcast before sowing",
            "best_season":         "Kharif (June-July)"
        },
        ('low',    'high',   'medium'): {
            "fertilizer":          "Urea",
            "npk_ratio":           "46:0:0",
            "application_method":  "Top dressing at vegetative stage",
            "best_season":         "Kharif and Rabi both"
        },
        ('low',    'high',   'high'):   {
            "fertilizer":          "Urea",
            "npk_ratio":           "46:0:0",
            "application_method":  "Split dose — 2 applications",
            "best_season":         "Rabi (October-November)"
        },
        ('medium', 'low',    'low'):    {
            "fertilizer":          "DAP + MOP",
            "npk_ratio":           "18:46:0 + 0:0:60",
            "application_method":  "Basal dose at time of sowing",
            "best_season":         "Kharif (June-July)"
        },
        ('medium', 'low',    'medium'): {
            "fertilizer":          "SSP (Single Super Phosphate)",
            "npk_ratio":           "0:16:0",
            "application_method":  "Mix into soil 1 week before sowing",
            "best_season":         "Rabi (October-November)"
        },
        ('medium', 'low',    'high'):   {
            "fertilizer":          "DAP",
            "npk_ratio":           "18:46:0",
            "application_method":  "Basal application before sowing",
            "best_season":         "Kharif (June-July)"
        },
        ('medium', 'medium', 'low'):    {
            "fertilizer":          "MOP (Muriate of Potash)",
            "npk_ratio":           "0:0:60",
            "application_method":  "Broadcast and incorporate into soil",
            "best_season":         "Rabi (October-November)"
        },
        ('medium', 'medium', 'medium'): {
            "fertilizer":          "NPK 10:26:26",
            "npk_ratio":           "10:26:26",
            "application_method":  "Basal dose — mix well in soil",
            "best_season":         "Kharif and Rabi both"
        },
        ('medium', 'medium', 'high'):   {
            "fertilizer":          "Ammonium Sulphate + SSP",
            "npk_ratio":           "21:0:0 + 0:16:0",
            "application_method":  "Apply before irrigation",
            "best_season":         "Rabi (October-November)"
        },
        ('medium', 'high',   'low'):    {
            "fertilizer":          "MOP",
            "npk_ratio":           "0:0:60",
            "application_method":  "Side dressing near plant roots",
            "best_season":         "Kharif (June-July)"
        },
        ('medium', 'high',   'medium'): {
            "fertilizer":          "NPK Balanced 12:12:17",
            "npk_ratio":           "12:12:17",
            "application_method":  "Basal application at sowing",
            "best_season":         "Kharif and Rabi both"
        },
        ('medium', 'high',   'high'):   {
            "fertilizer":          "Ammonium Sulphate",
            "npk_ratio":           "21:0:0",
            "application_method":  "Top dressing after 3 weeks of sowing",
            "best_season":         "Rabi (October-November)"
        },
        ('high',   'low',    'low'):    {
            "fertilizer":          "DAP + MOP",
            "npk_ratio":           "18:46:0 + 0:0:60",
            "application_method":  "Basal dose before sowing",
            "best_season":         "Kharif (June-July)"
        },
        ('high',   'low',    'medium'): {
            "fertilizer":          "SSP",
            "npk_ratio":           "0:16:0",
            "application_method":  "Mix in soil 1 week before sowing",
            "best_season":         "Rabi (October-November)"
        },
        ('high',   'low',    'high'):   {
            "fertilizer":          "DAP",
            "npk_ratio":           "18:46:0",
            "application_method":  "Basal application only",
            "best_season":         "Kharif (June-July)"
        },
        ('high',   'medium', 'low'):    {
            "fertilizer":          "MOP",
            "npk_ratio":           "0:0:60",
            "application_method":  "Broadcast before last ploughing",
            "best_season":         "Kharif and Rabi both"
        },
        ('high',   'medium', 'medium'): {
            "fertilizer":          "No fertilizer needed",
            "npk_ratio":           "-",
            "application_method":  "Soil is nutrient rich — avoid over-fertilization",
            "best_season":         "-"
        },
        ('high',   'medium', 'high'):   {
            "fertilizer":          "No fertilizer needed",
            "npk_ratio":           "-",
            "application_method":  "Soil is highly fertile — no application needed",
            "best_season":         "-"
        },
        ('high',   'high',   'low'):    {
            "fertilizer":          "MOP",
            "npk_ratio":           "0:0:60",
            "application_method":  "Apply at time of sowing",
            "best_season":         "Rabi (October-November)"
        },
        ('high',   'high',   'medium'): {
            "fertilizer":          "No fertilizer needed",
            "npk_ratio":           "-",
            "application_method":  "Soil nutrients are sufficient",
            "best_season":         "-"
        },
        ('high',   'high',   'high'):   {
            "fertilizer":          "No fertilizer needed",
            "npk_ratio":           "-",
            "application_method":  "Soil is fully fertile — no fertilizer required",
            "best_season":         "-"
        },
    }

    result = FERTILIZER_DB.get(
        (n_lvl, p_lvl, k_lvl),
        {
            "fertilizer":         "NPK 10:10:10 Balanced",
            "npk_ratio":          "10:10:10",
            "application_method": "Basal application before sowing",
            "best_season":        "Kharif and Rabi both"
        }
    )

    return jsonify({
        "suitable_crop":       crop,
        "N":                   N,
        "P":                   P,
        "K":                   K,
        "recommended_fertilizer": result["fertilizer"],
        "npk_ratio":           result["npk_ratio"],
        "application_method":  result["application_method"],
        "best_season":         result["best_season"]
    })
# ─────────────────────────────────────────────
# Route 3: Health check
# ─────────────────────────────────────────────
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status":  "running",
        "model":   "Random Forest — 29 crops",
        "routes":  ["/predict", "/fertilizer"]
    })


if __name__ == '__main__':
    app.run(debug=True)