import pandas as pd
import sys

def process_data(input_path, output_path):
    print(f"Reading input data from {input_path}...")
    df = pd.read_csv(input_path)
    
    def parse_narration(row):
        narration = str(row['narration'])
        parts = narration.split('/')
        narr_type = parts[0]
        
        channel = 'UNKNOWN'
        recv_acc = 'UNKNOWN'
        recv_mob = None
        recv_name = None
        
        try:
            if narr_type == 'UPI':
                channel = 'UPI'
                recv_mob = parts[1] if len(parts) > 1 else None
                recv_name = parts[2] if len(parts) > 2 else None
                recv_acc = parts[3] if len(parts) > 3 else 'UNKNOWN'
            elif narr_type == 'IMPS':
                channel = 'IMPS'
                recv_acc = parts[1] if len(parts) > 1 else 'UNKNOWN'
                recv_name = parts[2] if len(parts) > 2 else None
            elif narr_type == 'WEB-TRF':
                channel = 'WEB'
                recv_acc = parts[1] if len(parts) > 1 else 'UNKNOWN'
                recv_name = parts[2] if len(parts) > 2 else None
            elif narr_type == 'APP-P2P':
                channel = 'APP'
                recv_mob = parts[1] if len(parts) > 1 else None
                recv_name = parts[2] if len(parts) > 2 else None
                recv_acc = parts[3] if len(parts) > 3 else 'UNKNOWN'
            elif narr_type == 'ATM-WDL':
                channel = 'ATM'
                recv_acc = str(row['account_number'])
            else:
                channel = 'UNKNOWN'
                recv_acc = 'UNKNOWN'
        except Exception as e:
            pass
            
        return pd.Series([channel, recv_acc, recv_mob, recv_name])

    print("Parsing narration...")
    # Apply parsing logic
    df[['channel', 'receiver_account', 'receiver_mobile', 'receiver_name']] = df.apply(parse_narration, axis=1)
    
    # Task 2: Add sender_account and ensure receiver_account existence
    df['sender_account'] = df['account_number']
    
    # Task 3: Edge cases — Ensure no missing receiver_account 
    # (ATM is handled in the parser; blanks are marked UNKNOWN but we can ensure it here)
    df['receiver_account'] = df['receiver_account'].fillna('UNKNOWN')
    
    print(f"Saving parsed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == '__main__':
    input_file = r"data/raw/transactions_v2.csv"
    output_file = r"data/processed/transactions_parsed.csv"
    process_data(input_file, output_file)
