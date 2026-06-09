import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def generate_synthetic_data(n_samples=5000):
    print(f"Generating {n_samples} synthetic loan applications...")
    np.random.seed(42)
    
    # 1. Demographics
    ages = np.random.randint(18, 70, n_samples)
    marital_statuses = np.random.choice(
        ['Single', 'Married', 'Divorced'],
        size=n_samples,
        p=[0.40, 0.45, 0.15]
    )
    
    # 2. Employment
    employment_types = np.random.choice(
        ['Salaried', 'Self-Employed', 'Student', 'Unemployed'],
        size=n_samples,
        p=[0.60, 0.25, 0.10, 0.05]
    )
    
    # 3. Years of Experience (correlated with Age and Employment)
    experience = []
    for i in range(n_samples):
        age = ages[i]
        emp = employment_types[i]
        if emp == 'Student':
            exp = 0
            # Students are typically younger
            ages[i] = np.random.randint(18, 26)
        elif emp == 'Unemployed':
            # Unemployed could have some experience
            exp = np.random.randint(0, max(1, age - 18 - 2)) if age > 20 else 0
        else: # Salaried or Self-Employed
            max_exp = max(0, age - 18 - 2) # Assume starting work at 20 at earliest
            exp = np.random.randint(0, min(max_exp + 1, 35))
        experience.append(exp)
    experience = np.array(experience)
    
    # 4. Existing Loans
    existing_loans = np.random.randint(0, 5, n_samples)
    
    # 5. Annual Income (correlated with Employment Type and Experience)
    annual_incomes = []
    for i in range(n_samples):
        emp = employment_types[i]
        exp = experience[i]
        age = ages[i]
        
        if emp == 'Student':
            income = np.random.randint(2000, 15000)
        elif emp == 'Unemployed':
            income = np.random.randint(0, 4000)
        else: # Salaried or Self-Employed
            base = 25000 + (exp * 3500) + ((age - 18) * 800)
            if emp == 'Self-Employed':
                # Higher variance for self-employed
                income = base + np.random.normal(5000, 25000)
            else:
                income = base + np.random.normal(0, 15000)
            income = max(12000, income)
        annual_incomes.append(int(income))
    annual_incomes = np.array(annual_incomes)
    
    # 6. Loan Amount (correlated with Income)
    loan_amounts = []
    for i in range(n_samples):
        income = annual_incomes[i]
        emp = employment_types[i]
        
        if emp in ['Student', 'Unemployed']:
            # Small loans
            amt = np.random.randint(1000, 10000)
        else:
            # Average loan is 35% of income, but ranges from 10% to 150%
            ratio = np.random.beta(2, 4) * 1.5
            amt = income * ratio
            amt = max(2000, min(250000, amt))
        loan_amounts.append(int(amt))
    loan_amounts = np.array(loan_amounts)
    
    # 7. Credit Score (correlated with income, existing loans, age, and employment)
    credit_scores = []
    for i in range(n_samples):
        age = ages[i]
        income = annual_incomes[i]
        loans = existing_loans[i]
        emp = employment_types[i]
        
        base = 580
        # Income positive contribution
        base += min(120, (income / 10000) * 6)
        # Age positive contribution
        base += min(80, (age - 18) * 1.5)
        # Deduct for loans
        base -= loans * 25
        # Employment modifiers
        if emp == 'Unemployed':
            base -= 60
        elif emp == 'Student':
            base -= 20
            
        score = base + np.random.normal(0, 45)
        score = max(300, min(850, score))
        credit_scores.append(int(score))
    credit_scores = np.array(credit_scores)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': ages,
        'AnnualIncome': annual_incomes,
        'LoanAmount': loan_amounts,
        'CreditScore': credit_scores,
        'EmploymentType': employment_types,
        'YearsOfExperience': experience,
        'ExistingLoans': existing_loans,
        'MaritalStatus': marital_statuses
    })
    
    # 8. Define Default Status (Target) using financial logic + noise
    default_probs = []
    for i in range(n_samples):
        row = df.iloc[i]
        dti = row['LoanAmount'] / max(1, row['AnnualIncome'])
        
        # Risk starts at 50%
        risk = 50.0
        
        # Credit Score influence (Heavy)
        cs = row['CreditScore']
        if cs >= 780:
            risk -= 35
        elif cs >= 720:
            risk -= 25
        elif cs >= 660:
            risk -= 10
        elif cs >= 600:
            risk += 5
        elif cs >= 500:
            risk += 25
        else:
            risk += 45 # Very high risk
            
        # Debt-to-Income (DTI) influence (Heavy)
        if dti > 1.2:
            risk += 40
        elif dti > 0.8:
            risk += 25
        elif dti > 0.4:
            risk += 15
        elif dti < 0.2:
            risk -= 15
            
        # Employment Type influence
        emp = row['EmploymentType']
        if emp == 'Unemployed':
            risk += 25
        elif emp == 'Student':
            risk += 15
        elif emp == 'Self-Employed':
            risk += 5
        else:
            risk -= 5
            
        # Experience influence
        exp = row['YearsOfExperience']
        if exp < 2:
            risk += 10
        elif exp > 8:
            risk -= 10
            
        # Existing Loans influence
        loans = row['ExistingLoans']
        if loans >= 3:
            risk += 15
        elif loans == 0:
            risk -= 5
            
        # Age influence (younger has slightly higher default rate)
        age = row['Age']
        if age < 25:
            risk += 5
        elif age > 55:
            risk -= 5
            
        # Add random noise to make it realistic
        noise = np.random.normal(0, 8)
        risk = max(0.0, min(100.0, risk + noise))
        default_probs.append(risk)
        
    df['DefaultProbability'] = default_probs
    # Binary target: 1 = Default (Risk score > 55), 0 = Safe (Risk score <= 55)
    df['Default'] = (df['DefaultProbability'] > 55.0).astype(int)
    
    # Save the synthetic dataset for documentation / audit page reference
    df.to_csv('synthetic_loan_data.csv', index=False)
    print("Dataset saved to synthetic_loan_data.csv")
    return df

def train_pipeline():
    df = generate_synthetic_data(5000)
    
    # Define features and target
    X = df.drop(columns=['DefaultProbability', 'Default'])
    y = df['Default']
    
    # Categorical and numerical columns
    num_cols = ['Age', 'AnnualIncome', 'LoanAmount', 'CreditScore', 'YearsOfExperience', 'ExistingLoans']
    cat_cols = ['EmploymentType', 'MaritalStatus']
    
    # Preprocessing pipelines
    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    # Full training pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42))
    ])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit the model
    print("Training Random Forest model...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importances extraction
    classifier = model_pipeline.named_steps['classifier']
    cat_encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    
    # Retrieve feature names after one-hot encoding
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + encoded_cat_cols
    
    importances = classifier.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop Feature Importances:")
    print(feature_importance_df.to_string(index=False))
    
    # Save the pipeline and dataset properties
    model_filename = 'model_pipeline.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"Model pipeline successfully saved to {model_filename}")
    
    # Save feature metadata for dashboard use
    metadata = {
        'accuracy': float(accuracy),
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'feature_importances': feature_importance_df.to_dict(orient='records'),
        'total_samples': len(df)
    }
    joblib.dump(metadata, 'model_metadata.joblib')
    print("Model metadata saved to model_metadata.joblib")

if __name__ == '__main__':
    train_pipeline()
