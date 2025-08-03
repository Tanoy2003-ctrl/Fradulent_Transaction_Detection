#!/usr/bin/env python3
"""
Simple Fraud Detection System
Usage: python fraud_checker.py
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class FraudDetector:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.model_path = '/Users/tanoys_mac/Desktop/Python_Projects/fraud_model.pkl'
        self.encoder_path = '/Users/tanoys_mac/Desktop/Python_Projects/label_encoder.pkl'
        self.scaler_path = '/Users/tanoys_mac/Desktop/Python_Projects/scaler.pkl'
        
    def train_model(self):
        """Train and save the fraud detection model"""
        print("🔄 Training fraud detection model...")
        
        try:
            # Load dataset
            df = pd.read_csv("/Users/tanoys_mac/Downloads/Fraud.csv")
            print(f"📊 Loaded dataset with {df.shape[0]:,} transactions")
            
            # Preprocessing
            self.label_encoder = LabelEncoder()
            df['type_encoded'] = self.label_encoder.fit_transform(df['type'])
            
            self.scaler = StandardScaler()
            df['amount_scaled'] = self.scaler.fit_transform(df[['amount']])
            
            # Prepare features
            columns_to_drop = ['nameOrig', 'nameDest', 'amount', 'type', 'isFlaggedFraud']
            df_processed = df.drop(columns_to_drop, axis=1)
            
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
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Save model and preprocessors
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print("✅ Model trained and saved successfully!")
            
        except FileNotFoundError:
            print("❌ Error: Fraud.csv file not found at /Users/tanoys_mac/Downloads/Fraud.csv")
            print("Please make sure the dataset file exists.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error training model: {e}")
            sys.exit(1)
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print("📚 Model not found. Training new model...")
            self.train_model()
    
    def predict_transaction(self, step, transaction_type, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest):
        """Predict if a transaction is fraudulent"""
        
        if self.model is None:
            self.load_model()
        
        # Validate transaction type
        valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        if transaction_type.upper() not in valid_types:
            raise ValueError(f"Invalid transaction type. Must be one of: {valid_types}")
        
        # Create input data
        transaction_data = {
            'step': step,
            'oldbalanceOrg': old_balance_orig,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'type_encoded': self.label_encoder.transform([transaction_type.upper()])[0],
            'amount_scaled': self.scaler.transform([[amount]])[0][0]
        }
        
        # Create DataFrame and predict
        df_input = pd.DataFrame([transaction_data])
        features = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_encoded', 'amount_scaled']
        X_input = df_input[features]
        
        prediction = self.model.predict(X_input)[0]
        probability = self.model.predict_proba(X_input)[0]
        
        return prediction, probability

def get_transaction_input():
    """Get transaction details from user"""
    print("\n" + "="*60)
    print("🔍 FRAUD DETECTION SYSTEM")
    print("="*60)
    print("Enter transaction details:")
    
    try:
        # Transaction type
        print("\nAvailable transaction types:")
        types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
        for i, t in enumerate(types, 1):
            print(f"{i}. {t}")
        
        type_choice = input("\nSelect transaction type (1-5): ").strip()
        type_mapping = {'1': 'PAYMENT', '2': 'TRANSFER', '3': 'CASH_OUT', '4': 'DEBIT', '5': 'CASH_IN'}
        transaction_type = type_mapping.get(type_choice, 'PAYMENT')
        
        # Other details
        step = int(input("Step (time unit, e.g., 1-743): "))
        amount = float(input("Transaction amount ($): "))
        old_balance_orig = float(input("Sender's old balance ($): "))
        new_balance_orig = float(input("Sender's new balance ($): "))
        old_balance_dest = float(input("Receiver's old balance ($): "))
        new_balance_dest = float(input("Receiver's new balance ($): "))
        
        return step, transaction_type, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest
        
    except ValueError:
        print("❌ Invalid input! Please enter numeric values where required.")
        return None

def display_result(transaction_type, amount, prediction, probability):
    """Display fraud detection result"""
    print("\n" + "="*60)
    print("🎯 FRAUD DETECTION RESULT")
    print("="*60)
    print(f"Transaction Type: {transaction_type}")
    print(f"Amount: ${amount:,.2f}")
    print("-"*60)
    
    if prediction == 1:
        print("🚨 FRAUD DETECTED! 🚨")
        print(f"🔴 Fraud Probability: {probability[1]:.1%}")
        print("⚠️  This transaction appears to be fraudulent!")
    else:
        print("✅ LEGITIMATE TRANSACTION")
        print(f"🟢 Fraud Probability: {probability[1]:.1%}")
        print("✓ This transaction appears to be legitimate.")
    
    print(f"🔵 Legitimate Probability: {probability[0]:.1%}")
    print("="*60)

def main():
    """Main function"""
    detector = FraudDetector()
    
    print("🛡️  Welcome to the Fraud Detection System!")
    print("This system uses machine learning to detect fraudulent transactions.")
    
    while True:
        print("\n" + "="*60)
        print("OPTIONS:")
        print("1. 🔍 Check a transaction for fraud")
        print("2. 🚪 Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == '1':
            transaction_data = get_transaction_input()
            if transaction_data:
                try:
                    step, transaction_type, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest = transaction_data
                    
                    print("\n🔄 Analyzing transaction...")
                    prediction, probability = detector.predict_transaction(
                        step, transaction_type, amount, old_balance_orig, 
                        new_balance_orig, old_balance_dest, new_balance_dest
                    )
                    
                    display_result(transaction_type, amount, prediction, probability)
                    
                except Exception as e:
                    print(f"❌ Error analyzing transaction: {e}")
        
        elif choice == '2':
            print("\n👋 Thank you for using the Fraud Detection System!")
            print("Stay safe and secure! 🛡️")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1 or 2.")

if __name__ == "__main__":
    main()