from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, abort
import pandas as pd
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secure_secret_key_here'  

# Load the trained model and preprocessing objects
model = joblib.load('vertigo_hybrid_xgb_lgb_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Feature columns and vertigo stages (unchanged)
feature_columns = [
    'Age', 'Gender', 'Dizziness', 'Lightheadedness', 'Nausea', 'Vomiting', 'Hearing_Loss',
    'Tinnitus', 'Ear_Pain', 'Headache', 'Blurred_Vision', 'Imbalance', 'Spinning_Sensation',
    'Sensitivity_to_Motion', 'History_of_Migraine', 'History_of_Ear_Infections',
    'History_of_Head_Trauma', 'Cardiovascular_Issues', 'Neurological_Issues',
    'Symptoms_After_Position_Change', 'Duration_of_Episodes', 'Frequency_of_Episodes',
    'Recent_Illness', 'Medication_Use', 'Stress_Level', 'Physical_Activity_Level',
    'Exposure_to_Loud_Noise'
]
categorical_cols = [col for col in feature_columns if col != 'Age']
numerical_cols = ['Age']

vertigo_stages = {
    "No Vertigo Detected": "Stage 0", "BPPV": "Stage 1", "BPPV (Benign Paroxysmal Positional Vertigo)": "Stage 1", 
    "Vestibular Migraine": "Stage 1","Vestibular Neuritis": "Stage 2", "Labyrinthitis": "Stage 2", "Other Vertigo Type": "Stage 2",
    "Ménière's Disease": "Stage 3", "Central Vertigo": "Stage 3"
}

# Simulated user database (replace with a real database in production)
users = {}

# Preprocessing and prediction functions (unchanged)
def preprocess_user_input(user_data):
    df = pd.DataFrame([user_data])
    for col in categorical_cols:
        if col in df.columns:
            try:
                le = label_encoders[col]
                if df[col].iloc[0] not in le.classes_:
                    return None, f"Invalid value for {col}: '{df[col].iloc[0]}' not recognized"
                df[col] = le.transform(df[col].astype(str))
            except Exception as e:
                return None, f"Invalid value for {col}: {str(e)}"
    if numerical_cols:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    return df[feature_columns], None

def predict_vertigo(user_data):
    processed_data, error = preprocess_user_input(user_data)
    if error:
        return None, None, None, error
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)
    predicted_class = label_encoders['Vertigo_Type'].inverse_transform(prediction)[0]
    class_probabilities = dict(zip(label_encoders['Vertigo_Type'].classes_, prediction_proba[0]))
    predicted_stage = vertigo_stages.get(predicted_class, "Unknown Stage")
    return predicted_class, predicted_stage, class_probabilities, None

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/test')
def test_page():
    return render_template('test.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.get_json()
        predicted_class, predicted_stage, class_probabilities, error = predict_vertigo(user_data)
        if error:
            return jsonify({'error': error}), 400
        response = {
            'predicted_vertigo_type': predicted_class,
            'stage': predicted_stage,
            'probabilities': {k: f"{v:.2%}" for k, v in class_probabilities.items()}
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"Contact Form: Name={name}, Email={email}, Message={message}")
        flash('Your response has been recorded. We’ll get back to you soon!', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')


@app.route('/health-information')
def health_information():
    return render_template('health-information.html')

@app.route('/hospitals')
def hospitals():
    return render_template('hospitals.html')

@app.route('/patient-care')
def patient_care():
    return render_template('patient-care.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('restricted'))
        else:
            flash('Invalid username or password.', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/restricted')
def restricted():
    if not session.get('logged_in'):
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    return render_template('loginsystem.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        if email in users:
            flash('Email already registered!', 'error')
            return redirect(url_for('signup'))
        users[email] = password
        flash('Sign up successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    email = request.form['email']
    if email in users:
        flash(f'A password reset link has been sent to {email}.', 'success')
    else:
        flash('Email not found.', 'error')
    return redirect(url_for('login'))

# Custom 404 error handler
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)