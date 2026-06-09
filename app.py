import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import joblib
import database as db
import model_explainer as explainer

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(24))

MODEL_PATH = 'model_pipeline.joblib'
METADATA_PATH = 'model_metadata.joblib'

# Self-healing check for model files
def get_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        print("Model assets not found. Training model inline...")
        import train_model
        train_model.train_pipeline()
    return joblib.load(MODEL_PATH), joblib.load(METADATA_PATH)

# Initialize DB on startup
with app.app_context():
    db.init_db()

# Helper function to check login
def is_logged_in():
    return 'user_id' in session

@app.context_processor
def inject_user():
    return {
        'logged_in': is_logged_in(),
        'user_name': session.get('user_name', ''),
        'user_email': session.get('user_email', '')
    }

# --- ROUTES ---

@app.route('/')
def home():
    # Return homepage
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email').strip()
        password = request.form.get('password')
        
        user = db.verify_user(email, password)
        if user:
            session['user_id'] = user['id']
            session['user_name'] = user['full_name']
            session['user_email'] = user['email']
            flash('Successfully logged in!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password. Please try again.', 'danger')
            
    return render_template('login.html', active_tab='login')

@app.route('/register', methods=['POST'])
def register():
    email = request.form.get('email').strip()
    password = request.form.get('password')
    full_name = request.form.get('full_name')
    
    success, message = db.register_user(email, password, full_name)
    if success:
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    else:
        flash(message, 'danger')
        return render_template('login.html', active_tab='register', register_email=email, register_name=full_name)

@app.route('/logout')
def logout():
    session.clear()
    flash('Successfully logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not is_logged_in():
        flash('Please login to access the Loan Prediction platform.', 'warning')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        try:
            # Gather inputs
            data = {
                'Age': int(request.form.get('age')),
                'AnnualIncome': float(request.form.get('annual_income')),
                'LoanAmount': float(request.form.get('loan_amount')),
                'CreditScore': int(request.form.get('credit_score')),
                'EmploymentType': request.form.get('employment_type'),
                'YearsOfExperience': int(request.form.get('experience_years')),
                'ExistingLoans': int(request.form.get('existing_loans')),
                'MaritalStatus': request.form.get('marital_status')
            }
            
            # Load ML model
            model, metadata = get_model()
            
            # Prepare data for model inference
            df_input = pd.DataFrame([data])
            
            # Predict default probability (class 1)
            pred_proba = model.predict_proba(df_input)[0][1] # probability of default
            risk_score = round(pred_proba * 100, 1)
            
            # Classify risk levels and recommendations
            if risk_score < 35.0:
                risk_level = 'Low Risk'
                recommendation = 'Approve (Highly Recommended)'
            elif risk_score < 65.0:
                risk_level = 'Medium Risk'
                recommendation = 'Approve with Conditions (Higher Interest / Co-signer)'
            else:
                risk_level = 'High Risk'
                recommendation = 'Reject (High Probability of Default)'
                
            result = {
                'risk_score': risk_score,
                'default_probability': risk_score,
                'risk_level': risk_level,
                'recommendation': recommendation
            }
            
            # Log in SQLite database
            prediction_id = db.insert_prediction(session['user_id'], data, result)
            
            if prediction_id:
                return redirect(url_for('results', prediction_id=prediction_id))
            else:
                flash('Error saving prediction results.', 'danger')
                
        except Exception as e:
            flash(f"Error executing machine learning prediction: {str(e)}", 'danger')
            
    return render_template('predict.html')

@app.route('/results/<int:prediction_id>')
def results(prediction_id):
    if not is_logged_in():
        flash('Please login to view results.', 'warning')
        return redirect(url_for('login'))
        
    prediction = db.get_prediction_by_id(prediction_id, session['user_id'])
    if not prediction:
        flash('Prediction record not found.', 'danger')
        return redirect(url_for('history'))
        
    # Translate DB row to dictionary for explainer
    applicant_data = {
        'Age': prediction['age'],
        'AnnualIncome': prediction['annual_income'],
        'LoanAmount': prediction['loan_amount'],
        'CreditScore': prediction['credit_score'],
        'EmploymentType': prediction['employment_type'],
        'YearsOfExperience': prediction['experience_years'],
        'ExistingLoans': prediction['existing_loans'],
        'MaritalStatus': prediction['marital_status']
    }
    
    # Calculate DTI for display
    dti_ratio = round((prediction['loan_amount'] / max(1.0, prediction['annual_income'])) * 100, 1)
    
    # Generate XAI factors
    xai_factors = explainer.explain_risk(applicant_data)
    
    return render_template(
        'results.html', 
        prediction=prediction, 
        dti_ratio=dti_ratio,
        xai_factors=xai_factors
    )

@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        flash('Please login to view the analytics dashboard.', 'warning')
        return redirect(url_for('login'))
        
    summary = db.get_analytics_summary(session['user_id'])
    return render_template('dashboard.html', summary=summary)

@app.route('/history')
def history():
    if not is_logged_in():
        flash('Please login to view your history.', 'warning')
        return redirect(url_for('login'))
        
    records = db.get_prediction_history(session['user_id'])
    return render_template('history.html', records=records)

@app.route('/about')
def about():
    # Fetch model metadata (accuracy, feature importances, etc.)
    model, metadata = get_model()
    return render_template('about.html', metadata=metadata)

# --- API ENDPOINTS ---

@app.route('/api/analytics-data')
def api_analytics_data():
    if not is_logged_in():
        return jsonify({'error': 'Unauthorized'}), 401
    
    summary = db.get_analytics_summary(session['user_id'])
    return jsonify(summary)

if __name__ == '__main__':
    # Start web server
    app.run(debug=True, host='0.0.0.0', port=5000)
