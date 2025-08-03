#!/usr/bin/env python3
"""
Test script to demonstrate using fraud.py as an imported module
"""

import sys
import os
sys.path.append('/Users/tanoys_mac/Desktop/Python_Projects')

from Fraud import check_fraud, check_fraud_simple, print_fraud_result, check_fraud_batch

def test_individual_transactions():
    """Test individual transaction checking"""
    print("🧪 Testing Individual Transactions")
    print("="*50)
    
    # Test 1: Legitimate payment
    print("\n1. Testing legitimate payment:")
    result = check_fraud_simple('PAYMENT', 500, 2000, 1500)
    print(f"   Result: {'FRAUD' if result['is_fraud'] else 'LEGITIMATE'} ({result['fraud_probability']:.1%})")
    
    # Test 2: Suspicious transfer
    print("\n2. Testing suspicious transfer:")
    result = check_fraud(
        step=100,
        transaction_type='TRANSFER',
        amount=80000,
        sender_old_balance=80000,
        sender_new_balance=0,
        receiver_old_balance=0,
        receiver_new_balance=0
    )
    print_fraud_result(result)

def test_batch_processing():
    """Test batch transaction processing"""
    print("\n🧪 Testing Batch Processing")
    print("="*50)
    
    transactions = [
        {
            'transaction_type': 'PAYMENT',
            'amount': 150,
            'sender_old_balance': 1000,
            'sender_new_balance': 850
        },
        {
            'step': 200,
            'transaction_type': 'CASH_OUT',
            'amount': 30000,
            'sender_old_balance': 30000,
            'sender_new_balance': 0,
            'receiver_old_balance': 0,
            'receiver_new_balance': 30000
        },
        {
            'transaction_type': 'DEBIT',
            'amount': 2500,
            'sender_old_balance': 5000,
            'sender_new_balance': 2500
        }
    ]
    
    results = check_fraud_batch(transactions)
    
    for result in results:
        if 'error' not in result:
            tx_id = result['transaction_id']
            status = 'FRAUD' if result['is_fraud'] else 'LEGITIMATE'
            prob = result['fraud_probability']
            risk = result['risk_level']
            print(f"Transaction {tx_id}: {status} | Risk: {risk} | Prob: {prob:.1%}")
        else:
            print(f"Transaction {result['transaction_id']}: ERROR - {result['error']}")

def test_different_transaction_types():
    """Test different transaction types"""
    print("\n🧪 Testing Different Transaction Types")
    print("="*50)
    
    test_cases = [
        ('PAYMENT', 100, 'Small payment'),
        ('TRANSFER', 25000, 'Medium transfer'),
        ('CASH_OUT', 5000, 'Cash withdrawal'),
        ('DEBIT', 1500, 'Debit transaction'),
        ('CASH_IN', 2000, 'Cash deposit')
    ]
    
    for tx_type, amount, description in test_cases:
        result = check_fraud_simple(tx_type, amount, amount + 1000, 1000)
        status = 'FRAUD' if result['is_fraud'] else 'LEGITIMATE'
        prob = result['fraud_probability']
        print(f"{description:15} ({tx_type:8}): {status:10} | Fraud: {prob:5.1%}")

if __name__ == "__main__":
    print("🛡️  FRAUD DETECTION API TESTING")
    print("="*60)
    print("This script demonstrates how to use fraud.py as an imported module")
    
    try:
        test_individual_transactions()
        test_batch_processing()
        test_different_transaction_types()
        
        print("\n✅ All tests completed successfully!")
        print("\n📚 Usage Summary:")
        print("   - Use check_fraud() for full transaction analysis")
        print("   - Use check_fraud_simple() for quick checks")
        print("   - Use check_fraud_batch() for multiple transactions")
        print("   - Use print_fraud_result() for formatted output")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")