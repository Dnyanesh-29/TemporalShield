import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

ACCOUNT_POOL = [f"ACC_{i:05d}" for i in range(10000, 10050)] # Pool of 50 accounts

# Define realistic roles, their allowed account types, shift hours, and normal daily volume
ROLES_CONFIG = {
    'loan_officer': {
        'accounts': ['loan', 'mortgage'],
        'hours': (8, 18), # 8 AM to 6 PM
        'daily_vol': (20, 50)
    },
    'savings_representative': {
        'accounts': ['savings', 'checking'],
        'hours': (8, 18),
        'daily_vol': (40, 80)
    },
    'branch_manager': {
        'accounts': ['loan', 'mortgage', 'savings', 'checking', 'business'],
        'hours': (7, 19),
        'daily_vol': (10, 30)
    },
    'it_admin': {
        'accounts': ['system', 'audit'],
        'hours': (0, 24), # 24/7 access allowed
        'daily_vol': (5, 20)
    },
    'compliance_officer': {
        'accounts': ['loan', 'mortgage', 'savings', 'checking', 'business', 'system', 'audit'],
        'hours': (9, 17),
        'daily_vol': (50, 100)
    }
}

# Action types and their relative probabilities (normal behavior)
ACTIONS = ['read', 'write', 'transfer', 'override']
ACTION_WEIGHTS = [0.85, 0.10, 0.04, 0.01] 

def generate_access_logs(num_days=30, num_employees=50, output_file='access_logs.csv'):
    print(f"Generating synthetic access logs for {num_days} days...")
    
    # 1. Generate master list of employees
    employees = []
    roles_list = list(ROLES_CONFIG.keys())
    for i in range(num_employees):
        employees.append({
            'employee_id': f"EMP_{i+1:04d}",
            'role': random.choice(roles_list),
            'branch_code': f"BR_{random.randint(1, 10):03d}"
        })
    
    logs = []
    # Start date 30 days ago
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=num_days)
    
    # 2. Generate Normal Activity & Clean High-Volume Scenario
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        
        # Clean Scenario: High-volume legitimate day (e.g., month-end closing on day 25)
        is_busy_day = (day == 25)
        
        for emp in employees:
            role = emp['role']
            config = ROLES_CONFIG[role]
            
            vol_min, vol_max = config['daily_vol']
            
            # If it's a busy day, double the normal volume for customer-facing roles
            if is_busy_day and role in ['savings_representative', 'loan_officer', 'branch_manager']:
                vol_max = int(vol_max * 2.5)
                vol_min = int(vol_min * 2.0)
                
            daily_events = random.randint(vol_min, vol_max)
            
            for _ in range(daily_events):
                # Pick a random hour within their allowed shift
                h_start, h_end = config['hours']
                hour = random.randint(h_start, h_end - 1)
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                timestamp = current_date.replace(hour=hour, minute=minute, second=second)
                
                account_type = random.choice(config['accounts'])
                account_id = random.choice(ACCOUNT_POOL)
                action = random.choices(ACTIONS, weights=ACTION_WEIGHTS)[0]
                
                # Normal read is 1-5 records, writes/transfers are 1
                records = random.randint(1, 5) if action == 'read' else 1
                
                logs.append({
                    'employee_id': emp['employee_id'],
                    'role': role,
                    'account_id': account_id,
                    'account_type': account_type,
                    'branch_code': emp['branch_code'],
                    'action_type': action,
                    'timestamp': timestamp,
                    'records_accessed': records,
                    'is_suspicious': 0
                })
                
    # 3. Plant Suspicious Scenarios (Anomalies)
    
    # Scenario A: Temporal Link
    # Generate 10 instances spread across the month
    for day_offset in [2, 5, 8, 11, 15, 19, 22, 25, 27, 29]:
        emp_a = random.choice([e for e in employees if e['role'] == 'savings_representative'])
        date_a = start_date + timedelta(days=day_offset, hours=10, minutes=15)
        acc_a = random.choice(ACCOUNT_POOL)
        
        # The access event (Normally shouldn't be flagged in access logs, only in pairs)
        logs.append({
            'employee_id': emp_a['employee_id'],
            'role': emp_a['role'],
            'account_id': acc_a,
            'account_type': 'savings',
            'branch_code': emp_a['branch_code'],
            'action_type': 'read',
            'timestamp': date_a,
            'records_accessed': 1,
            'is_suspicious': 0
        })
        
        # The transfer 4 minutes later
        logs.append({
            'employee_id': emp_a['employee_id'],
            'role': emp_a['role'],
            'account_id': acc_a,
            'account_type': 'savings',
            'branch_code': emp_a['branch_code'],
            'action_type': 'transfer',
            'timestamp': date_a + timedelta(minutes=4),
            'records_accessed': 1,
            'is_suspicious': 0
        })

    # Scenario B: Volume Spike (Employee suddenly accesses 50+ accounts in one day)
    # Generate 5 spikes for loan_officer on different days
    for day_offset in [3, 9, 14, 21, 28]:
        emp_b = random.choice([e for e in employees if e['role'] == 'loan_officer'])
        date_b = start_date + timedelta(days=day_offset)
        for i in range(50):
            hour = random.randint(9, 16)
            minute = random.randint(0, 59)
            ts = date_b.replace(hour=hour, minute=minute, second=random.randint(0, 59))
            logs.append({
                'employee_id': emp_b['employee_id'],
                'role': emp_b['role'],
                'account_id': random.choice(ACCOUNT_POOL),
                'account_type': 'loan',
                'branch_code': emp_b['branch_code'],
                'action_type': 'read',
                'timestamp': ts,
                'records_accessed': random.randint(10, 50), 
                'is_suspicious': 1
            })
            
    # Scenario C: Off-Hours + Role Mismatch (Access outside role scope at 2 AM)
    # Generate instances for branch_manager, compliance_officer, savings_rep
    for day_offset, anom_role in [(4, 'savings_representative'), (12, 'branch_manager'), (18, 'compliance_officer'), (26, 'savings_representative')]:
        emp_c = random.choice([e for e in employees if e['role'] == anom_role])
        date_c = start_date + timedelta(days=day_offset, hours=2, minutes=14)
        
        for i in range(4):
            logs.append({
                'employee_id': emp_c['employee_id'],
                'role': emp_c['role'],
                'account_id': random.choice(ACCOUNT_POOL),
                'account_type': 'business' if anom_role != 'branch_manager' else 'system', 
                'branch_code': emp_c['branch_code'],
                'action_type': 'override',
                'timestamp': date_c + timedelta(minutes=i*3),
                'records_accessed': 1,
                'is_suspicious': 1
            })

    # 4. Compile, Sort, and Save
    df = pd.DataFrame(logs)
    
    # Sort chronologically to simulate a real log file
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Success! Generated {len(df)} records.")
    print(f"Saved to: {output_file}")
    
    # Print a quick distribution summary
    print("\nRole Distribution:")
    print(df['role'].value_counts())
    print("\nSuspicious Events:")
    print(df['is_suspicious'].value_counts())

if __name__ == "__main__":
    # Get directory of this script to ensure CSV is saved in the data folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, 'access_logs.csv')
    
    generate_access_logs(num_days=30, num_employees=50, output_file=output_path)
