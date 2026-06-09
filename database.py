import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_NAME = 'smartloan.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they do not exist."""
    print("Initializing SQLite database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            age INTEGER NOT NULL,
            annual_income REAL NOT NULL,
            loan_amount REAL NOT NULL,
            credit_score INTEGER NOT NULL,
            employment_type TEXT NOT NULL,
            experience_years INTEGER NOT NULL,
            existing_loans INTEGER NOT NULL,
            marital_status TEXT NOT NULL,
            risk_score REAL NOT NULL,
            default_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database tables initialized successfully.")

# User authentication CRUD
def register_user(email, password, full_name):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        cursor.execute(
            'INSERT INTO users (email, password_hash, full_name) VALUES (?, ?, ?)',
            (email, password_hash, full_name)
        )
        conn.commit()
        conn.close()
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Email already registered."
    except Exception as e:
        return False, f"Database error: {str(e)}"

def verify_user(email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        return {
            'id': user['id'],
            'email': user['email'],
            'full_name': user['full_name']
        }
    return None

# Prediction CRUD
def insert_prediction(user_id, data, result):
    """
    data: dict of inputs (age, annual_income, etc.)
    result: dict of outputs (risk_score, default_probability, risk_level, recommendation)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                user_id, age, annual_income, loan_amount, credit_score, 
                employment_type, experience_years, existing_loans, marital_status,
                risk_score, default_probability, risk_level, recommendation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            int(data['Age']),
            float(data['AnnualIncome']),
            float(data['LoanAmount']),
            int(data['CreditScore']),
            data['EmploymentType'],
            int(data['YearsOfExperience']),
            int(data['ExistingLoans']),
            data['MaritalStatus'],
            float(result['risk_score']),
            float(result['default_probability']),
            result['risk_level'],
            result['recommendation']
        ))
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return prediction_id
    except Exception as e:
        print(f"Error logging prediction: {str(e)}")
        return None

def get_prediction_history(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_prediction_by_id(prediction_id, user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions 
        WHERE id = ? AND user_id = ?
    ''', (prediction_id, user_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# Analytics metrics helpers
def get_analytics_summary(user_id):
    """Fetch aggregates for KPIs and charts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. KPI Counts
    cursor.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ?', (user_id,))
    total_apps = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND risk_level IN ('Low Risk', 'Medium Risk')", (user_id,))
    approved_loans = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = ? AND risk_level = 'High Risk'", (user_id,))
    rejected_loans = cursor.fetchone()[0] or 0
    
    cursor.execute('SELECT AVG(risk_score) FROM predictions WHERE user_id = ?', (user_id,))
    avg_risk = cursor.fetchone()[0] or 0.0
    
    # 2. Risk Distribution (Low, Medium, High)
    cursor.execute('''
        SELECT risk_level, COUNT(*) as count 
        FROM predictions 
        WHERE user_id = ? 
        GROUP BY risk_level
    ''', (user_id,))
    risk_dist = {row['risk_level']: row['count'] for row in cursor.fetchall()}
    
    # 3. Monthly Prediction counts (Trends)
    # Using strftime to get YYYY-MM
    cursor.execute('''
        SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count
        FROM predictions 
        WHERE user_id = ?
        GROUP BY month 
        ORDER BY month ASC
        LIMIT 6
    ''', (user_id,))
    trends = [{'month': row['month'], 'count': row['count']} for row in cursor.fetchall()]
    
    # 4. Average Risk by Employment Type
    cursor.execute('''
        SELECT employment_type, AVG(risk_score) as avg_score
        FROM predictions
        WHERE user_id = ?
        GROUP BY employment_type
    ''', (user_id,))
    emp_risk = [{'employment_type': row['employment_type'], 'avg_score': round(row['avg_score'], 1)} for row in cursor.fetchall()]

    # 5. Recent Predictions
    cursor.execute('''
        SELECT id, created_at, credit_score, loan_amount, risk_score, risk_level 
        FROM predictions 
        WHERE user_id = ?
        ORDER BY created_at DESC 
        LIMIT 5
    ''', (user_id,))
    recent = [dict(row) for row in cursor.fetchall()]

    conn.close()
    
    return {
        'kpis': {
            'total': total_apps,
            'approved': approved_loans,
            'rejected': rejected_loans,
            'avg_risk': round(avg_risk, 1)
        },
        'risk_distribution': {
            'Low Risk': risk_dist.get('Low Risk', 0),
            'Medium Risk': risk_dist.get('Medium Risk', 0),
            'High Risk': risk_dist.get('High Risk', 0)
        },
        'trends': trends,
        'employment_risk': emp_risk,
        'recent': recent
    }

if __name__ == '__main__':
    init_db()
