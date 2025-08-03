"""
Simple Fraud Detection API
Easy-to-use functions for fraud detection
"""

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Global variables to store model components
_model = None
_label_encoder = None
_scaler = None
_model_loaded = False

def _load_model():
    """Load the trained model and preprocessors"""
    global _model, _label_encoder, _scaler, _model_loaded
    
    if _model_loaded:
        return
    
    model_path = '/Users/tanoys_mac/Desktop/Python_Projects/fraud_model.pkl'
    encoder_path = '/Users/tanoys_mac/Desktop/Python_Projects/label_encoder.pkl'
    scaler_path = '/Users/tanoys_mac/Desktop/Python_Projects/scaler.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            _model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            _label_encoder = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            _scaler = pickle.load(f)
        _model_loaded = True
        print("✅ Fraud detection model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model files not found. Please train the model first using fraud_checker.py")
        raise

def check_fraud(step, transaction_type, amount, sender_old_balance, sender_new_balance, receiver_old_balance, receiver_new_balance):
    """
    Check if a transaction is fraudulent
    
    Parameters:
    - step: Time step (integer, e.g., 1-743)
    - transaction_type: Type of transaction ('PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN')
    - amount: Transaction amount (float)
    - sender_old_balance: Sender's balance before transaction (float)
    - sender_new_balance: Sender's balance after transaction (float)
    - receiver_old_balance: Receiver's balance before transaction (float)
    - receiver_new_balance: Receiver's balance after transaction (float)
    
    Returns:
    - Dictionary with prediction results
    """
    
    # Load model if not already loaded
    if not _model_loaded:
        _load_model()
    
    # Validate inputs
    valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    transaction_type = transaction_type.upper()
    if transaction_type not in valid_types:
        raise ValueError(f"Invalid transaction type. Must be one of: {valid_types}")
    
    # Prepare input data
    try:
        transaction_data = {
            'step': int(step),
            'oldbalanceOrg': float(sender_old_balance),
            'newbalanceOrig': float(sender_new_balance),
            'oldbalanceDest': float(receiver_old_balance),
            'newbalanceDest': float(receiver_new_balance),
            'type_encoded': _label_encoder.transform([transaction_type])[0],
            'amount_scaled': _scaler.transform([[float(amount)]])[0][0]
        }
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid input values: {e}")
    
    # Create DataFrame and make prediction
    df_input = pd.DataFrame([transaction_data])
    features = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_encoded', 'amount_scaled']
    X_input = df_input[features]
    
    prediction = _model.predict(X_input)[0]
    probability = _model.predict_proba(X_input)[0]
    
    # Return results
    result = {
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability[1]),
        'legitimate_probability': float(probability[0]),
        'transaction_type': transaction_type,
        'amount': float(amount),
        'confidence': 'HIGH' if max(probability) > 0.8 else 'MEDIUM' if max(probability) > 0.6 else 'LOW'
    }
    
    return result

def check_fraud_simple(transaction_type, amount, sender_old_balance, sender_new_balance):
    """
    Simplified fraud check with minimal parameters
    
    Parameters:
    - transaction_type: Type of transaction
    - amount: Transaction amount
    - sender_old_balance: Sender's balance before transaction
    - sender_new_balance: Sender's balance after transaction
    
    Returns:
    - Dictionary with prediction results
    """
    return check_fraud(
        step=1,
        transaction_type=transaction_type,
        amount=amount,
        sender_old_balance=sender_old_balance,
        sender_new_balance=sender_new_balance,
        receiver_old_balance=0,
        receiver_new_balance=0
    )

def print_fraud_result(result):
    """Pretty print fraud detection result"""
    print("\n" + "="*50)
    print("🎯 FRAUD DETECTION RESULT")
    print("="*50)
    print(f"Transaction Type: {result['transaction_type']}")
    print(f"Amount: ${result['amount']:,.2f}")
    print(f"Confidence: {result['confidence']}")
    print("-"*50)
    
    if result['is_fraud']:
        print("🚨 FRAUD DETECTED! 🚨")
        print(f"🔴 Fraud Probability: {result['fraud_probability']:.1%}")
    else:
        print("✅ LEGITIMATE TRANSACTION")
        print(f"🟢 Fraud Probability: {result['fraud_probability']:.1%}")
    
    print(f"🔵 Legitimate Probability: {result['legitimate_probability']:.1%}")
    print("="*50)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Fraud Detection API")
    print("="*40)
    
    # Test cases
    test_cases = [
        {
            'name': 'Small Payment',
            'params': {
                'step': 1,
                'transaction_type': 'PAYMENT',
                'amount': 100,
                'sender_old_balance': 1000,
                'sender_new_balance': 900,
                'receiver_old_balance': 0,
                'receiver_new_balance': 0
            }
        },
        {
            'name': 'Large Transfer',
            'params': {
                'step': 50,
                'transaction_type': 'TRANSFER',
                'amount': 50000,
                'sender_old_balance': 50000,
                'sender_new_balance': 0,
                'receiver_old_balance': 0,
                'receiver_new_balance': 0
            }
        },
        {
            'name': 'Cash Out',
            'params': {
                'step': 100,
                'transaction_type': 'CASH_OUT',
                'amount': 25000,
                'sender_old_balance': 25000,
                'sender_new_balance': 0,
                'receiver_old_balance': 0,
                'receiver_new_balance': 25000
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🔍 Testing: {test_case['name']}")
        try:
            result = check_fraud(**test_case['params'])
            print_fraud_result(result)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ API testing completed!")