def explain_risk(applicant_data):
    """
    Generate Explainable AI risk factors for a specific loan applicant.
    
    applicant_data: dict containing:
        - Age
        - AnnualIncome
        - LoanAmount
        - CreditScore
        - EmploymentType
        - YearsOfExperience
        - ExistingLoans
        - MaritalStatus
    
    Returns a list of dictionaries with:
        - feature: display name of feature
        - value: display value
        - status: 'danger' (red), 'warning' (yellow), 'success' (green)
        - message: human-readable explanation
        - score_impact: approximate direction (e.g. "+15% Risk", "-10% Risk")
    """
    factors = []
    
    # Extract features
    age = float(applicant_data['Age'])
    income = float(applicant_data['AnnualIncome'])
    loan = float(applicant_data['LoanAmount'])
    credit_score = float(applicant_data['CreditScore'])
    employment = applicant_data['EmploymentType']
    experience = float(applicant_data['YearsOfExperience'])
    existing_loans = int(applicant_data['ExistingLoans'])
    
    dti = loan / max(1.0, income)
    
    # 1. Credit Score Analysis
    if credit_score >= 750:
        factors.append({
            'feature': 'Credit Score',
            'value': f"{int(credit_score)} (Excellent)",
            'status': 'success',
            'message': 'Your credit score demonstrates excellent creditworthiness, significantly lowering default risk.',
            'score_impact': '-30% Risk'
        })
    elif credit_score >= 670:
        factors.append({
            'feature': 'Credit Score',
            'value': f"{int(credit_score)} (Good)",
            'status': 'success',
            'message': 'Your credit score is solid and fits within standard prime lending benchmarks.',
            'score_impact': '-15% Risk'
        })
    elif credit_score >= 580:
        factors.append({
            'feature': 'Credit Score',
            'value': f"{int(credit_score)} (Fair)",
            'status': 'warning',
            'message': 'Your credit score is fair. While acceptable, it introduces minor risk and may increase interest rates.',
            'score_impact': '+5% Risk'
        })
    else:
        factors.append({
            'feature': 'Credit Score',
            'value': f"{int(credit_score)} (Subprime)",
            'status': 'danger',
            'message': 'Your credit score is below 580. Subprime credit history is the leading indicator of potential loan defaults.',
            'score_impact': '+45% Risk'
        })
        
    # 2. Debt-to-Income (DTI) Analysis
    dti_percent = dti * 100
    if dti > 0.8:
        factors.append({
            'feature': 'Debt-to-Income (DTI)',
            'value': f"{dti_percent:.1f}%",
            'status': 'danger',
            'message': f"Extremely high DTI ratio. The loan amount represents {dti_percent:.1f}% of your annual income, indicating severe debt burden.",
            'score_impact': '+35% Risk'
        })
    elif dti > 0.4:
        factors.append({
            'feature': 'Debt-to-Income (DTI)',
            'value': f"{dti_percent:.1f}%",
            'status': 'warning',
            'message': f"Elevated DTI ratio ({dti_percent:.1f}%). The industry standard threshold for safe debt levels is 40% or below.",
            'score_impact': '+15% Risk'
        })
    else:
        factors.append({
            'feature': 'Debt-to-Income (DTI)',
            'value': f"{dti_percent:.1f}%",
            'status': 'success',
            'message': f"Excellent DTI ratio ({dti_percent:.1f}%). Your income comfortably covers the requested loan amount.",
            'score_impact': '-10% Risk'
        })
        
    # 3. Employment Type
    if employment == 'Unemployed':
        factors.append({
            'feature': 'Employment Status',
            'value': 'Unemployed',
            'status': 'danger',
            'message': 'Lack of current steady income represents a major default risk for any long-term credit facility.',
            'score_impact': '+30% Risk'
        })
    elif employment == 'Student':
        factors.append({
            'feature': 'Employment Status',
            'value': 'Student',
            'status': 'warning',
            'message': 'Student status indicates potential transition in income, adding moderate cash-flow risk.',
            'score_impact': '+15% Risk'
        })
    elif employment == 'Self-Employed':
        factors.append({
            'feature': 'Employment Status',
            'value': 'Self-Employed',
            'status': 'warning',
            'message': 'Self-employed income profiles are subject to higher cash-flow volatility than salaried employment.',
            'score_impact': '+5% Risk'
        })
    else: # Salaried
        factors.append({
            'feature': 'Employment Status',
            'value': 'Salaried (Full-Time)',
            'status': 'success',
            'message': 'Stable salaried employment offers predictable cash flow, reducing overall risk parameters.',
            'score_impact': '-5% Risk'
        })
        
    # 4. Years of Experience
    if experience < 2 and employment not in ['Student', 'Unemployed']:
        factors.append({
            'feature': 'Work Experience',
            'value': f"{int(experience)} Years",
            'status': 'warning',
            'message': 'Limited work history (<2 years) suggests early career phase or recent job transition, reflecting minor instability.',
            'score_impact': '+10% Risk'
        })
    elif experience >= 8:
        factors.append({
            'feature': 'Work Experience',
            'value': f"{int(experience)} Years",
            'status': 'success',
            'message': 'Extended employment experience (8+ years) signals stable career standing and persistent earning capability.',
            'score_impact': '-10% Risk'
        })
        
    # 5. Existing Loans
    if existing_loans >= 3:
        factors.append({
            'feature': 'Active Credit Lines',
            'value': f"{existing_loans} Loans",
            'status': 'danger',
            'message': f"Multiple active loans ({existing_loans}) indicate leverage risk and could stretch your monthly repayment capacity.",
            'score_impact': '+15% Risk'
        })
    elif existing_loans == 0:
        factors.append({
            'feature': 'Active Credit Lines',
            'value': 'No Existing Loans',
            'status': 'success',
            'message': 'No other active debt obligations. All disposable income is available to service this loan.',
            'score_impact': '-5% Risk'
        })
        
    # 6. Age Group
    if age < 25:
        factors.append({
            'feature': 'Applicant Age',
            'value': f"{int(age)} Years",
            'status': 'warning',
            'message': 'Younger demographic profiles typically correlate with shorter credit lines and higher variance in risk models.',
            'score_impact': '+5% Risk'
        })
    elif age > 55:
        factors.append({
            'feature': 'Applicant Age',
            'value': f"{int(age)} Years",
            'status': 'success',
            'message': 'Mature demographic bracket tends to correspond with highly structured asset profiles and lower risk metrics.',
            'score_impact': '-5% Risk'
        })

    # Sort factors by level of concern (danger first, then warning, then success)
    order = {'danger': 0, 'warning': 1, 'success': 2}
    factors.sort(key=lambda x: order[x['status']])
    
    return factors
