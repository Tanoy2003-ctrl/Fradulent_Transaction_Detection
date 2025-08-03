import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_and_save_model():
    """Train the fraud detection model and save it"""
    print("Training fraud detection model...")
    
    # Load dataset
    df = pd.read_csv("/Users/tanoys_mac/Downloads/Fraud.csv")
    
    # Preprocessing
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    
    scaler = StandardScaler()
    df['amount_scaled'] = scaler.fit_transform(df[['amount']])
    
    # Drop unnecessary columns
    columns_to_drop = ['nameOrig', 'nameDest', 'amount', 'type', 'isFlaggedFraud']
    df_processed = df.drop(columns_to_drop, axis=1)
    
    # Features and target
    X = df_processed.drop('isFraud', axis=1)
    y = df_processed['isFraud']
    
    # Handle class imbalance
    fraud = df_processed[df_processed['isFraud'] == 1]
    non_fraud = df_processed[df_processed['isFraud'] == 0].sample(n=len(fraud) * 5, random_state=42)
    df_balanced = pd.concat([fraud, non_fraud])
    
    X_balanced = df_balanced.drop('isFraud', axis=1)
    y_balanced = df_balanced['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and preprocessors
    with open('/Users/tanoys_mac/Desktop/Python_Projects/fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('/Users/tanoys_mac/Desktop/Python_Projects/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('/Users/tanoys_mac/Desktop/Python_Projects/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model trained and saved!")
    return model, le, scaler

def load_model():
    """Load the trained model"""
    try:
        with open('/Users/tanoys_mac/Desktop/Python_Projects/fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('/Users/tanoys_mac/Desktop/Python_Projects/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        with open('/Users/tanoys_mac/Desktop/Python_Projects/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, le, scaler
    except FileNotFoundError:
        return train_and_save_model()

def predict_fraud_from_input(step, transaction_type, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest):
    """Predict fraud from user input parameters"""
    
    # Load model
    model, le, scaler = load_model()
    
    # Create input data
    transaction_data = {
        'step': step,
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': old_balance_orig,
        'newbalanceOrig': new_balance_orig,
        'oldbalanceDest': old_balance_dest,
        'newbalanceDest': new_balance_dest
    }
    
    # Preprocess
    df_input = pd.DataFrame([transaction_data])
    df_input['type_encoded'] = le.transform([transaction_type])[0]
    df_input['amount_scaled'] = scaler.transform([[amount]])[0][0]
    
    # Select features
    features = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_encoded', 'amount_scaled']
    X_input = df_input[features]
    
    # Predict
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]
    
    return prediction, probability

def interactive_fraud_detection():
    """Interactive fraud detection system"""
    print("="*60)
    print("           FRAUD DETECTION SYSTEM")
    print("="*60)
    
    while True:
        print("\nEnter transaction details:")
        print("Available transaction types: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN")
        
        try:
            # Get user input
            step = int(input("Step (time unit, 1-743): "))
            transaction_type = input("Transaction type: ").upper().strip()
            amount = float(input("Amount: "))
            old_balance_orig = float(input("Sender's old balance: "))
            new_balance_orig = float(input("Sender's new balance: "))
            old_balance_dest = float(input("Receiver's old balance: "))
            new_balance_dest = float(input("Receiver's new balance: "))
            
            # Validate transaction type
            valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
            if transaction_type not in valid_types:
                print(f"Invalid transaction type. Using 'PAYMENT' as default.")
                transaction_type = 'PAYMENT'
            
            # Make prediction
            prediction, probability = predict_fraud_from_input(
                step, transaction_type, amount, old_balance_orig, 
                new_balance_orig, old_balance_dest, new_balance_dest
            )
            
            # Display results
            print("\n" + "="*60)
            print("FRAUD DETECTION RESULT")
            print("="*60)
            print(f"Transaction Type: {transaction_type}")
            print(f"Amount: ${amount:,.2f}")
            print(f"Step: {step}")
            print("-"*60)
            
            if prediction == 1:
                print("🚨 FRAUD DETECTED! 🚨")
                print(f"Fraud Probability: {probability[1]:.2%}")
            else:
                print("✅ LEGITIMATE TRANSACTION")
                print(f"Fraud Probability: {probability[1]:.2%}")
            
            print(f"Legitimate Probability: {probability[0]:.2%}")
            print("="*60)
            
        except ValueError:
            print("Invalid input! Please enter numeric values where required.")
        except Exception as e:
            print(f"Error: {e}")
        
        # Ask if user wants to continue
        continue_choice = input("\nCheck another transaction? (y/n): ").lower().strip()
        if continue_choice != 'y':
            break
    
    print("Thank you for using the Fraud Detection System!")

# Example usage with predefined inputs
def demo_with_examples():
    """Demo with example transactions"""
    print("="*60)
    print("           FRAUD DETECTION DEMO")
    print("="*60)
    
    # Example transactions
    examples = [
        {
            'name': 'Legitimate Payment',
            'step': 1,
            'type': 'PAYMENT',
            'amount': 1000.0,
            'oldbalanceOrg': 5000.0,
            'newbalanceOrig': 4000.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        },
        {
            'name': 'Suspicious Transfer',
            'step': 100,
            'type': 'TRANSFER',
            'amount': 50000.0,
            'oldbalanceOrg': 50000.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 0.0
        },
        {
            'name': 'Large Cash Out',
            'step': 200,
            'type': 'CASH_OUT',
            'amount': 100000.0,
            'oldbalanceOrg': 100000.0,
            'newbalanceOrig': 0.0,
            'oldbalanceDest': 0.0,
            'newbalanceDest': 100000.0
        }
    ]
    
    for example in examples:
        print(f"\nTesting: {example['name']}")
        print("-" * 40)
        
        prediction, probability = predict_fraud_from_input(
            example['step'], example['type'], example['amount'],
            example['oldbalanceOrg'], example['newbalanceOrig'],
            example['oldbalanceDest'], example['newbalanceDest']
        )
        
        print(f"Type: {example['type']}, Amount: ${example['amount']:,.2f}")
        if prediction == 1:
            print("🚨 FRAUD DETECTED!")
        else:
            print("✅ LEGITIMATE")
        print(f"Fraud Probability: {probability[1]:.2%}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Interactive fraud detection")
    print("2. Demo with examples")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        interactive_fraud_detection()
    elif choice == '2':
        demo_with_examples()
    else:
        print("Invalid choice. Running interactive mode...")
        interactive_fraud_detection()