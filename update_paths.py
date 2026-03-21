import os
import re

replacements = {
    r'"transactions_parsed\.csv"': r'"data/processed/transactions_parsed.csv"',
    r"'transactions_parsed\.csv'": r"'data/processed/transactions_parsed.csv'",
    r'"fraud_gnn_graph\.pt"': r'"outputs/visualizations/fraud_gnn_graph.pt"',
    r"'fraud_gnn_graph\.pt'": r"'outputs/visualizations/fraud_gnn_graph.pt'",
    r'"subgraph_viz\.png"': r'"outputs/visualizations/subgraph_viz.png"',
    r"'subgraph_viz\.png'": r"'outputs/visualizations/subgraph_viz.png'",
    r'"fraud_gnn_model\.pth"': r'"models/fraud_gnn_model.pth"',
    r"'fraud_gnn_model\.pth'": r"'models/fraud_gnn_model.pth'",
    r'"best_fraud_gnn_model\.pth"': r'"models/best_fraud_gnn_model.pth"',
    r"'best_fraud_gnn_model\.pth'": r"'models/best_fraud_gnn_model.pth'",
    r'"test_transactions\.csv"': r'"data/raw/test_transactions.csv"',
    r"'test_transactions\.csv'": r"'data/raw/test_transactions.csv'",
    r'"transactions\.csv"': r'"data/raw/transactions.csv"',
    r"'transactions\.csv'": r"'data/raw/transactions.csv'",
    r'"fraud_predictions\.csv"': r'"outputs/visualizations/fraud_predictions.csv"',
    r"'fraud_predictions\.csv'": r"'outputs/visualizations/fraud_predictions.csv'",
    r"'mule_network_complete\.png'": r"'outputs/visualizations/mule_accounts/mule_network_complete.png'",
    r'"mule_network_complete\.png"': r'"outputs/visualizations/mule_accounts/mule_network_complete.png"',
    r"'mule_accounts_analysis\.png'": r"'outputs/visualizations/mule_accounts/mule_accounts_analysis.png'",
    r'"mule_accounts_analysis\.png"': r'"outputs/visualizations/mule_accounts/mule_accounts_analysis.png'",
    r"'fraud_detection_report\.html'": r"'outputs/reports/fraud_detection_report.html'",
    r'"fraud_detection_report\.html"': r'"outputs/reports/fraud_detection_report.html'",
    r"'fraud_detection_analysis\.png'": r"'outputs/visualizations/fraud_detection_analysis.png'",
    r'"fraud_detection_analysis\.png"': r'"outputs/visualizations/fraud_detection_analysis.png'",
    r"'fraud_patterns_analysis\.png'": r"'outputs/visualizations/fraud_patterns_analysis.png'",
    r'"fraud_patterns_analysis\.png"': r'"outputs/visualizations/fraud_patterns_analysis.png'",
    r'c:\\Users\\SAKTHIVEL R\\Desktop\\DataSET\\transactions_v2\.csv': r'data/raw/transactions_v2.csv',
    r'c:\\Users\\SAKTHIVEL R\\Desktop\\DataSET\\transactions_parsed\.csv': r'data/processed/transactions_parsed.csv',
}

def replace_mule(m):
    return m.group(0).replace("mule_account", "outputs/visualizations/mule_accounts/mule_account")

dirs = ['src', 'scripts']

for d in dirs:
    if not os.path.exists(d): continue
    for root, _, files in os.walk(d):
        for file in files:
            if not file.endswith('.py'): continue
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = content
            for p, r in replacements.items():
                new_content = re.sub(p, r, new_content)
                
            new_content = re.sub(r"f'mule_account_[^']+\.png'", replace_mule, new_content)
            new_content = re.sub(r'f"mule_account_[^"]+\.png"', replace_mule, new_content)
                    
            if new_content != content:
                print(f"Updated {path}")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
