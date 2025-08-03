#!/usr/bin/env python3
"""
🛡️ COMPREHENSIVE FRAUD DETECTION SYSTEM
========================================

A complete machine learning-based fraud detection system that includes:
- Model training and evaluation
- Interactive command-line interface
- API functions for easy integration
- Batch processing capabilities
- Example usage demonstrations

Author: AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

# Global variables to store model components
_model = None
_label_encoder = None
_scaler = None
_model_loaded = False

# File paths
MODEL_PATH = '/Users/tanoys_mac/Desktop/Python_Projects/fraud_model.pkl'
ENCODER_PATH = '/Users/tanoys_mac/Desktop/Python_Projects/label_encoder.pkl'
SCALER_PATH = '/Users/tanoys_mac/Desktop/Python_Projects/scaler.pkl'
DATASET_PATH = '/Users/tanoys_mac/Downloads/Fraud.csv'

# ============================================================================
# MODEL TRAINING AND MANAGEMENT
# ============================================================================

def train_fraud_model(show_plots=False):
    """
    Train the fraud detection model from scratch
    
    Parameters:
    - show_plots: Whether to display training plots (default: False)
    
    Returns:
    - Tuple of (model, label_encoder, scaler)
    """
    print("🔄 Training fraud detection model...")
    print("="*60)
    
    try:
        # Load dataset
        print("📊 Loading dataset...")
        df = pd.read_csv(DATASET_PATH)
        print(f"✅ Loaded dataset with {df.shape[0]:,} transactions and {df.shape[1]} features")
        
        # Display dataset info
        print(f"📈 Dataset columns: {df.columns.tolist()}")
        target_col = 'isFraud'
        print(f"🎯 Target distribution:")
        print(df[target_col].value_counts())
        
        # Preprocessing
        print("\n🔧 Preprocessing data...")
        
        # Handle categorical variables
        le = LabelEncoder()
        df['type_encoded'] = le.fit_transform(df['type'])
        print(f"✅ Encoded transaction types: {le.classes_}")
        
        # Scale the amount column
        scaler = StandardScaler()
        df['amount_scaled'] = scaler.fit_transform(df[['amount']])
        print("✅ Scaled transaction amounts")
        
        # Drop unnecessary columns for modeling
        columns_to_drop = ['nameOrig', 'nameDest', 'amount', 'type', 'isFlaggedFraud']
        df_processed = df.drop(columns_to_drop, axis=1)
        print(f"✅ Processed dataset shape: {df_processed.shape}")
        
        # Features and target
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
        
        print(f"📊 Features: {X.columns.tolist()}")
        print(f"📊 Feature matrix shape: {X.shape}")
        
        # Handle class imbalance using undersampling
        print("\n⚖️ Handling class imbalance...")
        fraud = df_processed[df_processed[target_col] == 1]
        non_fraud = df_processed[df_processed[target_col] == 0].sample(n=len(fraud) * 5, random_state=42)
        df_balanced = pd.concat([fraud, non_fraud])
        
        print(f"✅ Balanced dataset shape: {df_balanced.shape}")
        print("✅ Balanced target distribution:")
        print(df_balanced[target_col].value_counts())
        
        X_balanced = df_balanced.drop(target_col, axis=1)
        y_balanced = df_balanced[target_col]
        
        # Train-test split
        X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42
        )
        
        # Train model
        print("\n🤖 Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_bal, y_train_bal)
        print("✅ Model training completed!")
        
        # Evaluate model
        print("\n📊 Evaluating model performance...")
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        y_pred = model.predict(X_test_orig)
        y_pred_proba = model.predict_proba(X_test_orig)[:, 1]
        
        roc_score = roc_auc_score(y_test_orig, y_pred_proba)
        print(f"🎯 ROC AUC Score: {roc_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Top 5 Feature Importances:")
        for idx, row in feature_importance.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save model and preprocessors
        print("\n💾 Saving model and preprocessors...")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(ENCODER_PATH, 'wb') as f:
            pickle.dump(le, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        print("✅ Model, encoder, and scaler saved successfully!")
        
        # Display plots if requested
        if show_plots:
            display_training_plots(df, feature_importance)
        
        return model, le, scaler
        
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at {DATASET_PATH}")
        print("Please make sure the Fraud.csv file exists at the specified location.")
        raise
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        raise

def load_fraud_model():
    """Load the trained model and preprocessors"""
    global _model, _label_encoder, _scaler, _model_loaded
    
    if _model_loaded:
        return _model, _label_encoder, _scaler
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            _label_encoder = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
        _model_loaded = True
        print("✅ Fraud detection model loaded successfully!")
        return _model, _label_encoder, _scaler
    except FileNotFoundError:
        print("📚 Model files not found. Training new model...")
        model, le, scaler = train_fraud_model()
        _model, _label_encoder, _scaler = model, le, scaler
        _model_loaded = True
        return model, le, scaler

def display_training_plots(df, feature_importance):
    """Display training visualization plots"""
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Target distribution
        plt.subplot(2, 3, 1)
        df['isFraud'].value_counts().plot(kind='bar')
        plt.title('Fraud vs Legitimate Transactions')
        plt.xlabel('Transaction Type')
        plt.ylabel('Count')
        
        # Plot 2: Transaction types
        plt.subplot(2, 3, 2)
        df['type'].value_counts().plot(kind='bar')
        plt.title('Transaction Types Distribution')
        plt.xlabel('Transaction Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Plot 3: Amount distribution
        plt.subplot(2, 3, 3)
        plt.hist(df['amount'], bins=50, alpha=0.7)
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        # Plot 4: Feature importance
        plt.subplot(2, 3, 4)
        top_features = feature_importance.head(7)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.title('Top Feature Importances')
        plt.xlabel('Importance')
        
        # Plot 5: Fraud by transaction type
        plt.subplot(2, 3, 5)
        fraud_by_type = df.groupby('type')['isFraud'].mean()
        fraud_by_type.plot(kind='bar')
        plt.title('Fraud Rate by Transaction Type')
        plt.xlabel('Transaction Type')
        plt.ylabel('Fraud Rate')
        plt.xticks(rotation=45)
        
        # Plot 6: Amount vs Fraud
        plt.subplot(2, 3, 6)
        fraud_amounts = df[df['isFraud'] == 1]['amount']
        legit_amounts = df[df['isFraud'] == 0]['amount']
        plt.hist([legit_amounts, fraud_amounts], bins=50, alpha=0.7, 
                label=['Legitimate', 'Fraud'], color=['green', 'red'])
        plt.title('Amount Distribution: Fraud vs Legitimate')
        plt.xlabel('Amount')
        plt.ylabel('Frequency')
        plt.legend()
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not display plots: {e}")

# ============================================================================
# FRAUD DETECTION API FUNCTIONS
# ============================================================================

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
    model, le, scaler = load_fraud_model()
    
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
            'type_encoded': le.transform([transaction_type])[0],
            'amount_scaled': scaler.transform([[float(amount)]])[0][0]
        }
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid input values: {e}")
    
    # Create DataFrame and make prediction
    df_input = pd.DataFrame([transaction_data])
    features = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_encoded', 'amount_scaled']
    X_input = df_input[features]
    
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]
    
    # Return results
    result = {
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability[1]),
        'legitimate_probability': float(probability[0]),
        'transaction_type': transaction_type,
        'amount': float(amount),
        'step': int(step),
        'confidence': 'HIGH' if max(probability) > 0.8 else 'MEDIUM' if max(probability) > 0.6 else 'LOW',
        'risk_level': 'CRITICAL' if probability[1] > 0.9 else 'HIGH' if probability[1] > 0.7 else 'MEDIUM' if probability[1] > 0.3 else 'LOW'
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

def check_fraud_batch(transactions):
    """
    Check multiple transactions for fraud
    
    Parameters:
    - transactions: List of transaction dictionaries
    
    Returns:
    - List of prediction results
    """
    results = []
    for i, tx in enumerate(transactions):
        try:
            if 'step' in tx:
                result = check_fraud(**tx)
            else:
                result = check_fraud_simple(**tx)
            result['transaction_id'] = i
            results.append(result)
        except Exception as e:
            results.append({
                'transaction_id': i,
                'error': str(e),
                'is_fraud': None
            })
    return results

def print_fraud_result(result):
    """Pretty print fraud detection result"""
    print("\n" + "="*60)
    print("🎯 FRAUD DETECTION RESULT")
    print("="*60)
    print(f"Transaction Type: {result['transaction_type']}")
    print(f"Amount: ${result['amount']:,.2f}")
    if 'step' in result:
        print(f"Step: {result['step']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Risk Level: {result['risk_level']}")
    print("-"*60)
    
    if result['is_fraud']:
        print("🚨 FRAUD DETECTED! 🚨")
        print(f"🔴 Fraud Probability: {result['fraud_probability']:.1%}")
        if result['risk_level'] == 'CRITICAL':
            print("⚠️  CRITICAL RISK - Immediate action required!")
        elif result['risk_level'] == 'HIGH':
            print("⚠️  HIGH RISK - Review recommended!")
    else:
        print("✅ LEGITIMATE TRANSACTION")
        print(f"🟢 Fraud Probability: {result['fraud_probability']:.1%}")
        print("✓ Transaction appears to be legitimate.")
    
    print(f"🔵 Legitimate Probability: {result['legitimate_probability']:.1%}")
    print("="*60)

# ============================================================================
# INTERACTIVE USER INTERFACE
# ============================================================================

def get_transaction_input():
    """Get transaction details from user input"""
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
        
        return {
            'step': step,
            'transaction_type': transaction_type,
            'amount': amount,
            'sender_old_balance': old_balance_orig,
            'sender_new_balance': new_balance_orig,
            'receiver_old_balance': old_balance_dest,
            'receiver_new_balance': new_balance_dest
        }
        
    except ValueError:
        print("❌ Invalid input! Please enter numeric values where required.")
        return None
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return None

def interactive_fraud_detection():
    """Interactive fraud detection system"""
    print("🛡️  Welcome to the Interactive Fraud Detection System!")
    print("This system uses machine learning to detect fraudulent transactions.")
    
    while True:
        print("\n" + "="*60)
        print("OPTIONS:")
        print("1. 🔍 Check a single transaction")
        print("2. 📊 Run demo with examples")
        print("3. 🔧 Train new model")
        print("4. 📈 Show model statistics")
        print("5. 🚪 Exit")
        print("="*60)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            transaction_data = get_transaction_input()
            if transaction_data:
                try:
                    print("\n🔄 Analyzing transaction...")
                    result = check_fraud(**transaction_data)
                    print_fraud_result(result)
                except Exception as e:
                    print(f"❌ Error analyzing transaction: {e}")
        
        elif choice == '2':
            run_demo_examples()
        
        elif choice == '3':
            try:
                print("\n🔄 Training new model...")
                train_fraud_model(show_plots=False)
                print("✅ Model training completed!")
            except Exception as e:
                print(f"❌ Error training model: {e}")
        
        elif choice == '4':
            show_model_statistics()
        
        elif choice == '5':
            print("\n👋 Thank you for using the Fraud Detection System!")
            print("Stay safe and secure! 🛡️")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-5.")

def run_demo_examples():
    """Run demonstration with example transactions"""
    print("\n" + "="*60)
    print("🎮 FRAUD DETECTION DEMO")
    print("="*60)
    
    examples = [
        {
            'name': 'Legitimate Small Payment',
            'step': 1,
            'transaction_type': 'PAYMENT',
            'amount': 250.50,
            'sender_old_balance': 2000,
            'sender_new_balance': 1749.50,
            'receiver_old_balance': 0,
            'receiver_new_balance': 0
        },
        {
            'name': 'Suspicious Large Transfer',
            'step': 100,
            'transaction_type': 'TRANSFER',
            'amount': 75000,
            'sender_old_balance': 75000,
            'sender_new_balance': 0,
            'receiver_old_balance': 0,
            'receiver_new_balance': 0
        },
        {
            'name': 'Fraudulent Cash Out',
            'step': 200,
            'transaction_type': 'CASH_OUT',
            'amount': 50000,
            'sender_old_balance': 50000,
            'sender_new_balance': 0,
            'receiver_old_balance': 0,
            'receiver_new_balance': 50000
        },
        {
            'name': 'Normal ATM Withdrawal',
            'step': 50,
            'transaction_type': 'CASH_OUT',
            'amount': 200,
            'sender_old_balance': 1000,
            'sender_new_balance': 800,
            'receiver_old_balance': 0,
            'receiver_new_balance': 200
        },
        {
            'name': 'Large Debit Transaction',
            'step': 150,
            'transaction_type': 'DEBIT',
            'amount': 25000,
            'sender_old_balance': 30000,
            'sender_new_balance': 5000,
            'receiver_old_balance': 0,
            'receiver_new_balance': 0
        }
    ]
    
    for example in examples:
        print(f"\n🔍 Testing: {example['name']}")
        print("-" * 40)
        
        try:
            result = check_fraud(**{k: v for k, v in example.items() if k != 'name'})
            
            status = "🚨 FRAUD" if result['is_fraud'] else "✅ LEGITIMATE"
            risk = result['risk_level']
            prob = result['fraud_probability']
            
            print(f"Type: {example['transaction_type']}, Amount: ${example['amount']:,.2f}")
            print(f"Result: {status} | Risk: {risk} | Fraud Prob: {prob:.1%}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Demo completed! Tested {len(examples)} transactions.")

def show_model_statistics():
    """Display model statistics and information"""
    try:
        model, le, scaler = load_fraud_model()
        
        print("\n" + "="*60)
        print("📊 MODEL STATISTICS")
        print("="*60)
        
        print(f"🤖 Model Type: Random Forest Classifier")
        print(f"🌳 Number of Trees: {model.n_estimators}")
        print(f"🎯 Model Classes: {model.classes_}")
        print(f"📊 Number of Features: {model.n_features_in_}")
        
        print(f"\n🏷️  Transaction Types Supported:")
        for i, type_name in enumerate(le.classes_):
            print(f"   {i}: {type_name}")
        
        print(f"\n📈 Feature Importance (Top 5):")
        feature_names = ['step', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_encoded', 'amount_scaled']
        importances = model.feature_importances_
        
        # Sort features by importance
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance in feature_importance[:5]:
            print(f"   {feature}: {importance:.3f}")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error loading model statistics: {e}")

# ============================================================================
# BATCH PROCESSING AND UTILITIES
# ============================================================================

def process_csv_file(csv_path, output_path=None):
    """
    Process a CSV file with transactions and add fraud predictions
    
    Parameters:
    - csv_path: Path to input CSV file
    - output_path: Path to save results (optional)
    
    Returns:
    - DataFrame with fraud predictions
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Processing {len(df)} transactions from {csv_path}")
        
        results = []
        for idx, row in df.iterrows():
            try:
                result = check_fraud(
                    step=row.get('step', 1),
                    transaction_type=row['type'],
                    amount=row['amount'],
                    sender_old_balance=row['oldbalanceOrg'],
                    sender_new_balance=row['newbalanceOrig'],
                    receiver_old_balance=row['oldbalanceDest'],
                    receiver_new_balance=row['newbalanceDest']
                )
                results.append({
                    'transaction_id': idx,
                    'is_fraud_predicted': result['is_fraud'],
                    'fraud_probability': result['fraud_probability'],
                    'risk_level': result['risk_level'],
                    'confidence': result['confidence']
                })
            except Exception as e:
                results.append({
                    'transaction_id': idx,
                    'is_fraud_predicted': None,
                    'fraud_probability': None,
                    'risk_level': 'ERROR',
                    'confidence': 'ERROR',
                    'error': str(e)
                })
        
        # Combine original data with predictions
        results_df = pd.DataFrame(results)
        final_df = pd.concat([df, results_df], axis=1)
        
        if output_path:
            final_df.to_csv(output_path, index=False)
            print(f"✅ Results saved to {output_path}")
        
        # Summary statistics
        fraud_count = sum(1 for r in results if r.get('is_fraud_predicted') == True)
        total_count = len([r for r in results if r.get('is_fraud_predicted') is not None])
        
        print(f"📊 Summary: {fraud_count}/{total_count} transactions flagged as fraud ({fraud_count/total_count*100:.1f}%)")
        
        return final_df
        
    except Exception as e:
        print(f"❌ Error processing CSV file: {e}")
        return None

# ============================================================================
# MAIN EXECUTION AND COMMAND LINE INTERFACE
# ============================================================================

def print_help():
    """Print help information"""
    print("""
🛡️  FRAUD DETECTION SYSTEM - HELP
================================

USAGE:
    python fraud.py [command] [options]

COMMANDS:
    interactive     Start interactive fraud detection mode
    train          Train a new fraud detection model
    demo           Run demonstration with example transactions
    api            Run API examples and tests
    help           Show this help message

EXAMPLES:
    python fraud.py interactive    # Start interactive mode
    python fraud.py train         # Train new model
    python fraud.py demo          # Run demo examples
    python fraud.py api           # Test API functions

API USAGE:
    from fraud import check_fraud, check_fraud_simple, print_fraud_result
    
    # Full fraud check
    result = check_fraud(step=100, transaction_type='TRANSFER', amount=50000,
                        sender_old_balance=50000, sender_new_balance=0,
                        receiver_old_balance=0, receiver_new_balance=0)
    
    # Simplified fraud check
    result = check_fraud_simple('PAYMENT', 1000, 5000, 4000)
    
    # Print results
    print_fraud_result(result)

TRANSACTION TYPES:
    PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN

For more information, visit the documentation or run in interactive mode.
""")

def run_api_tests():
    """Run API tests and examples"""
    print("🧪 FRAUD DETECTION API TESTS")
    print("="*60)
    
    # Test cases
    test_cases = [
        {
            'name': 'Legitimate Small Payment',
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
            'name': 'Suspicious Large Transfer',
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
            'name': 'Fraudulent Cash Out',
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
    
    # Test simplified API
    print(f"\n🔍 Testing Simplified API:")
    try:
        result = check_fraud_simple('DEBIT', 5000, 10000, 5000)
        print_fraud_result(result)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ API testing completed!")

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        interactive_fraud_detection()
    else:
        command = sys.argv[1].lower()
        
        if command == 'interactive':
            interactive_fraud_detection()
        elif command == 'train':
            try:
                train_fraud_model(show_plots=True)
            except Exception as e:
                print(f"❌ Training failed: {e}")
        elif command == 'demo':
            run_demo_examples()
        elif command == 'api':
            run_api_tests()
        elif command == 'help':
            print_help()
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python fraud.py help' for usage information.")

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Stay safe and secure! 🛡️")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your input and try again.")