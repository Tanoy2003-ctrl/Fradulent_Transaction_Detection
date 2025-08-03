"""
Example usage of the Fraud Detection System
This file shows different ways to use the fraud detection API
"""

from fraud_api import check_fraud, check_fraud_simple, print_fraud_result

def example_1_basic_usage():
    """Example 1: Basic fraud detection"""
    print("="*60)
    print("EXAMPLE 1: Basic Fraud Detection")
    print("="*60)
    
    # Check a suspicious transaction
    result = check_fraud(
        step=100,
        transaction_type='TRANSFER',
        amount=75000,
        sender_old_balance=75000,
        sender_new_balance=0,
        receiver_old_balance=0,
        receiver_new_balance=0
    )
    
    print_fraud_result(result)

def example_2_legitimate_transaction():
    """Example 2: Legitimate transaction"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Legitimate Transaction")
    print("="*60)
    
    # Check a normal payment
    result = check_fraud(
        step=5,
        transaction_type='PAYMENT',
        amount=250.50,
        sender_old_balance=2000,
        sender_new_balance=1749.50,
        receiver_old_balance=0,
        receiver_new_balance=0
    )
    
    print_fraud_result(result)

def example_3_simplified_check():
    """Example 3: Using simplified function"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Simplified Check")
    print("="*60)
    
    # Use simplified function for quick checks
    result = check_fraud_simple(
        transaction_type='CASH_OUT',
        amount=15000,
        sender_old_balance=15000,
        sender_new_balance=0
    )
    
    print_fraud_result(result)

def example_4_batch_checking():
    """Example 4: Check multiple transactions"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Transaction Checking")
    print("="*60)
    
    transactions = [
        {
            'name': 'Online Purchase',
            'step': 10,
            'type': 'PAYMENT',
            'amount': 89.99,
            'sender_old': 500,
            'sender_new': 410.01,
            'receiver_old': 0,
            'receiver_new': 0
        },
        {
            'name': 'Large Wire Transfer',
            'step': 200,
            'type': 'TRANSFER',
            'amount': 100000,
            'sender_old': 100000,
            'sender_new': 0,
            'receiver_old': 0,
            'receiver_new': 0
        },
        {
            'name': 'ATM Withdrawal',
            'step': 50,
            'type': 'CASH_OUT',
            'amount': 200,
            'sender_old': 1000,
            'sender_new': 800,
            'receiver_old': 0,
            'receiver_new': 200
        }
    ]
    
    for transaction in transactions:
        print(f"\n🔍 Checking: {transaction['name']}")
        result = check_fraud(
            step=transaction['step'],
            transaction_type=transaction['type'],
            amount=transaction['amount'],
            sender_old_balance=transaction['sender_old'],
            sender_new_balance=transaction['sender_new'],
            receiver_old_balance=transaction['receiver_old'],
            receiver_new_balance=transaction['receiver_new']
        )
        
        # Simple result display
        status = "🚨 FRAUD" if result['is_fraud'] else "✅ LEGITIMATE"
        confidence = result['fraud_probability']
        print(f"   Result: {status} (Fraud: {confidence:.1%})")

def example_5_interactive():
    """Example 5: Interactive fraud checking"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Interactive Fraud Checking")
    print("="*60)
    
    print("Enter transaction details to check for fraud:")
    
    try:
        transaction_type = input("Transaction type (PAYMENT/TRANSFER/CASH_OUT/DEBIT/CASH_IN): ").upper()
        amount = float(input("Amount: $"))
        sender_old = float(input("Sender's old balance: $"))
        sender_new = float(input("Sender's new balance: $"))
        
        result = check_fraud_simple(transaction_type, amount, sender_old, sender_new)
        print_fraud_result(result)
        
    except ValueError:
        print("❌ Invalid input! Please enter numeric values for amounts.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🛡️  FRAUD DETECTION SYSTEM - USAGE EXAMPLES")
    print("This demonstrates different ways to use the fraud detection API")
    
    # Run all examples
    example_1_basic_usage()
    example_2_legitimate_transaction()
    example_3_simplified_check()
    example_4_batch_checking()
    
    # Ask if user wants to try interactive mode
    print("\n" + "="*60)
    try_interactive = input("Would you like to try interactive mode? (y/n): ").lower().strip()
    if try_interactive == 'y':
        example_5_interactive()
    
    print("\n🎉 Examples completed! You can now use the fraud detection system in your own code.")
    print("\n📚 Quick Reference:")
    print("   from fraud_api import check_fraud, check_fraud_simple, print_fraud_result")
    print("   result = check_fraud(step, type, amount, old_bal, new_bal, recv_old, recv_new)")
    print("   print_fraud_result(result)")