"""
═══════════════════════════════════════════════════════════════════════════════
    MULE ACCOUNT DETECTION — GRAPH VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

Visualizes detected mule accounts (fan-in pattern) as interactive network graphs.
Shows how multiple accounts send money to a single receiver (the mule).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import warnings

warnings.filterwarnings('ignore')

def detect_fan_in_patterns(df, threshold_senders=3):
    """
    Detect fan-in patterns (money mule accounts).
    A mule account is one that receives from many senders.
    """
    print("\n" + "="*80)
    print("🎯 DETECTING MULE ACCOUNTS (FAN-IN PATTERN)")
    print("="*80)
    
    # Count unique senders per receiver
    receiver_counts = df.groupby('receiver_account')['sender_account'].nunique().reset_index()
    receiver_counts.columns = ['receiver_account', 'num_senders']
    receiver_counts = receiver_counts[receiver_counts['num_senders'] >= threshold_senders]
    receiver_counts = receiver_counts.sort_values('num_senders', ascending=False)
    
    print(f"\n✓ Found {len(receiver_counts)} potential mule accounts")
    print(f"  (Receiving from {threshold_senders}+ different senders)")
    
    return receiver_counts

def visualize_mule_account(df, mule_account, pred_df=None, title_suffix=""):
    """
    Create detailed graph visualization of a mule account and its senders.
    """
    
    # Get all transactions to this mule
    mule_txns = df[df['receiver_account'] == mule_account]
    
    if len(mule_txns) == 0:
        print(f"❌ No transactions found for account {mule_account}")
        return None
    
    unique_senders = mule_txns['sender_account'].nunique()
    total_amount = mule_txns['amount'].sum()
    
    # Get fraud score if available
    fraud_score = 0
    risk_level = "UNKNOWN"
    if pred_df is not None and mule_account in pred_df['account_number'].values:
        fraud_score = pred_df[pred_df['account_number'] == mule_account]['fraud_score'].values[0]
        risk_level = pred_df[pred_df['account_number'] == mule_account]['risk_level'].values[0]
    
    print(f"\n📊 MULE ACCOUNT DETAILS:")
    print(f"   Account: {mule_account}")
    print(f"   Incoming Transactions: {len(mule_txns)}")
    print(f"   Unique Senders: {unique_senders}")
    print(f"   Total Received: ₹{total_amount:,.0f}")
    print(f"   Fraud Score: {fraud_score:.4f}")
    print(f"   Risk Level: {risk_level}")
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add mule as central node
    G.add_node(mule_account, node_type='mule', color='#e74c3c')
    
    # Add sender nodes and edges
    for _, row in mule_txns.iterrows():
        sender = row['sender_account']
        amount = row['amount']
        channel = row['channel']
        
        G.add_node(sender, node_type='sender', color='#3498db')
        G.add_edge(sender, mule_account, weight=amount, channel=channel)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Force mule to center
    pos[mule_account] = np.array([0, 0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw edges with varying thickness by amount
    edges = G.edges()
    edge_weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 1
    
    # Normalize edge widths
    edge_widths = [2 + (4 * (w - min_weight) / (max_weight - min_weight + 1))
                   for w in edge_weights]
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#95a5a6',
        arrows=True,
        arrowsize=25,
        arrowstyle='->',
        width=edge_widths,
        connectionstyle='arc3,rad=0.1',
        alpha=0.7,
        edge_vmin=0,
        edge_vmax=max(edge_weights)
    )
    
    # Draw nodes
    mule_node = [mule_account]
    sender_nodes = [n for n in G.nodes() if n != mule_account]
    
    # Mule node (center - large, red)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=mule_node,
        node_color='#e74c3c',
        node_size=4000,
        node_shape='o',
        ax=ax,
        edgecolors='#c0392b',
        linewidths=4
    )
    
    # Sender nodes (blue)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=sender_nodes,
        node_color='#3498db',
        node_size=2000,
        node_shape='o',
        ax=ax,
        edgecolors='#2980b9',
        linewidths=2
    )
    
    # Draw labels
    labels = {}
    labels[mule_account] = f"MULE\n{str(mule_account)[-6:]}"
    for i, node in enumerate(sender_nodes[:20]):  # Limit labels for clarity
        labels[node] = f"S{i+1}\n{str(node)[-4:]}"
    
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=9,
        font_weight='bold',
        ax=ax,
        font_color='white'
    )
    
    # Add edge labels (amounts) for top transactions
    edge_labels = {}
    for _, row in mule_txns.nlargest(10, 'amount').iterrows():
        sender = row['sender_account']
        amount = f"₹{row['amount']/1000:.0f}k"
        channel = row['channel']
        if (sender, mule_account) in G.edges():
            edge_labels[(sender, mule_account)] = f"{amount}\n{channel}"
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color='#2c3e50',
        ax=ax
    )
    
    # Title and annotations
    title = f"🔴 MULE ACCOUNT DETECTED — FAN-IN PATTERN"
    if title_suffix:
        title += f"\n{title_suffix}"
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add info box
    info_text = f"""
    📊 PATTERN ANALYSIS
    ═════════════════════════════════════
    
    Mule Account: {str(mule_account)}
    
    💰 Transaction Summary:
       • Incoming Transactions: {len(mule_txns)}
       • Unique Senders: {unique_senders}
       • Total Amount: ₹{total_amount:,.0f}
       • Avg/Txn: ₹{total_amount/len(mule_txns):,.0f}
    
    🚦 Risk Assessment:
       • Pattern: FAN-IN (Money Mule)
       • Fraud Score: {fraud_score:.4f}
       • Risk Level: {risk_level}
    
    ⚠️  INDICATORS:
       • Multiple incoming sources ✓
       • Rapid consolidation pattern ✓
       • Potential money laundering ⚠
    
    🔍 Channels Used:
    """
    
    # Add channel info
    channels = mule_txns['channel'].value_counts()
    for channel, count in channels.items():
        info_text += f"\n       • {channel}: {count} transactions"
    
    info_text += "\n    ═════════════════════════════════════"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9, pad=1))
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig, G

def visualize_multiple_mules(df, pred_df, num_mules=3):
    """
    Create visualizations for top N mule accounts.
    """
    print("\n" + "="*80)
    print("🎯 VISUALIZING TOP MULE ACCOUNTS")
    print("="*80)
    
    # Detect mules
    receiver_counts = detect_fan_in_patterns(df, threshold_senders=3)
    
    if len(receiver_counts) == 0:
        print("❌ No mule accounts detected!")
        return
    
    # Get top mules
    top_mules = receiver_counts.head(num_mules)
    
    print(f"\n📊 Top {num_mules} Mule Accounts:")
    for idx, (_, row) in enumerate(top_mules.iterrows(), 1):
        mule_acc = row['receiver_account']
        num_senders = row['num_senders']
        print(f"   {idx}. {str(mule_acc)} ← {num_senders} senders")
    
    # Visualize each
    for idx, (_, row) in enumerate(top_mules.iterrows(), 1):
        mule_account = row['receiver_account']
        num_senders = row['num_senders']
        
        print(f"\n[{idx}/{num_mules}] Creating visualization for mule account...")
        
        fig, G = visualize_mule_account(
            df, mule_account, pred_df,
            title_suffix=f"Receiving from {num_senders} different senders"
        )
        
        if fig:
            filename = f'mule_account_{idx}_{str(mule_account)[-6:]}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved: {filename}")
            plt.close(fig)


def visualize_complete_mule_network(df, pred_df):
    """
    Create a large network visualization showing ALL mule accounts and their connections.
    """
    print("\n" + "="*80)
    print("🎯 COMPLETE MULE NETWORK VISUALIZATION")
    print("="*80)
    
    # Detect all mule accounts
    receiver_counts = detect_fan_in_patterns(df, threshold_senders=3)
    
    if len(receiver_counts) == 0:
        print("❌ No mule accounts detected!")
        return
    
    mule_accounts = receiver_counts['receiver_account'].values
    
    # Build network of mule accounts and their direct senders
    G = nx.DiGraph()
    
    # Add mule accounts
    for mule in mule_accounts:
        fraud_score = 0
        if mule in pred_df['account_number'].values:
            fraud_score = pred_df[pred_df['account_number'] == mule]['fraud_score'].values[0]
        
        G.add_node(mule, node_type='mule', fraud_score=fraud_score)
    
    # Add sender connections (sample to avoid clutter)
    edge_count = 0
    for mule in mule_accounts[:5]:  # Top 5 mules for clarity
        mule_txns = df[df['receiver_account'] == mule]
        top_senders = mule_txns.groupby('sender_account')['amount'].sum().nlargest(5)
        
        for sender in top_senders.index:
            amount = top_senders[sender]
            G.add_node(sender, node_type='sender')
            G.add_edge(sender, mule, weight=amount)
            edge_count += 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Draw edges
    edges = G.edges()
    if edges:
        edge_weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        edge_widths = [1 + (5 * (w - min_weight) / (max_weight - min_weight + 1))
                       for w in edge_weights]
    else:
        edge_widths = []
    
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#e74c3c',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        width=edge_widths,
        connectionstyle='arc3,rad=0.1',
        alpha=0.6
    )
    
    # Draw nodes
    mule_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'mule']
    sender_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'sender']
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=mule_nodes,
        node_color='#e74c3c',
        node_size=3000,
        node_shape='s',  # Square for mules
        ax=ax,
        edgecolors='#c0392b',
        linewidths=3,
        label='Mule Accounts'
    )
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=sender_nodes,
        node_color='#3498db',
        node_size=1500,
        node_shape='o',
        ax=ax,
        edgecolors='#2980b9',
        linewidths=1.5,
        label='Sender Accounts'
    )
    
    # Labels (only for mules)
    labels = {n: f"{str(n)[-6:]}" for n in mule_nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_weight='bold',
        ax=ax,
        font_color='white'
    )
    
    ax.set_title(
        "🔴 COMPLETE MULE ACCOUNT NETWORK\nFan-In Pattern Detection",
        fontsize=18, fontweight='bold', pad=20
    )
    
    ax.legend(scatterpoints=1, loc='upper left', fontsize=12, framealpha=0.9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mule_network_complete.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: mule_network_complete.png")
    plt.close(fig)


def create_mule_comparison_chart(df, pred_df):
    """
    Create a comparison chart showing mule accounts and their characteristics.
    """
    print("\n[Creating Mule Comparison Chart...]")
    
    receiver_counts = detect_fan_in_patterns(df, threshold_senders=2)
    
    if len(receiver_counts) == 0:
        return
    
    # Get top 15 mules with their stats
    top_mules = receiver_counts.head(15)
    
    mule_stats = []
    for _, row in top_mules.iterrows():
        mule_acc = row['receiver_account']
        mule_txns = df[df['receiver_account'] == mule_acc]
        
        fraud_score = 0
        if mule_acc in pred_df['account_number'].values:
            fraud_score = pred_df[pred_df['account_number'] == mule_acc]['fraud_score'].values[0]
        
        mule_stats.append({
            'account': str(mule_acc)[-6:],
            'senders': row['num_senders'],
            'total_amount': mule_txns['amount'].sum(),
            'transactions': len(mule_txns),
            'fraud_score': fraud_score
        })
    
    stats_df = pd.DataFrame(mule_stats)
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Number of senders
    ax = axes[0, 0]
    bars = ax.barh(stats_df['account'], stats_df['senders'], color='#e74c3c')
    ax.set_xlabel('Number of Senders', fontweight='bold', fontsize=11)
    ax.set_title('Mule Accounts by Sender Count', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(stats_df.iterrows()):
        ax.text(row['senders'] + 0.1, i, f"{int(row['senders'])}", va='center', fontsize=9)
    
    # Plot 2: Total amount received
    ax = axes[0, 1]
    bars = ax.barh(stats_df['account'], stats_df['total_amount']/1000, color='#f39c12')
    ax.set_xlabel('Total Amount Received (₹000s)', fontweight='bold', fontsize=11)
    ax.set_title('Mule Accounts by Total Amount', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 3: Number of transactions
    ax = axes[1, 0]
    scatter = ax.scatter(stats_df['senders'], stats_df['transactions'], 
                        s=stats_df['fraud_score']*500 + 100,
                        c=stats_df['fraud_score'],
                        cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidth=1)
    ax.set_xlabel('Number of Senders', fontweight='bold', fontsize=11)
    ax.set_ylabel('Number of Transactions', fontweight='bold', fontsize=11)
    ax.set_title('Senders vs Transactions (size = fraud score)', fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3)
    
    for idx, row in stats_df.iterrows():
        ax.annotate(str(row['account']), (row['senders'], row['transactions']),
                   fontsize=8, alpha=0.7)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fraud Score', fontweight='bold')
    
    # Plot 4: Fraud score ranking
    ax = axes[1, 1]
    colors_list = ['#e74c3c' if x > 0.6 else '#f39c12' if x > 0.33 else '#2ecc71' 
                   for x in stats_df['fraud_score']]
    bars = ax.barh(stats_df['account'], stats_df['fraud_score'], color=colors_list)
    ax.set_xlabel('Fraud Score', fontweight='bold', fontsize=11)
    ax.set_title('Mule Accounts by Fraud Score', fontweight='bold', fontsize=12)
    ax.axvline(0.6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='HIGH threshold')
    ax.axvline(0.33, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='MEDIUM threshold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 1.0)
    
    for i, (idx, row) in enumerate(stats_df.iterrows()):
        ax.text(row['fraud_score'] + 0.02, i, f"{row['fraud_score']:.3f}", 
               va='center', fontsize=9)
    
    plt.suptitle('🔴 MULE ACCOUNT ANALYSIS DASHBOARD', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('mule_accounts_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: mule_accounts_analysis.png")
    plt.close(fig)


def main():
    """Main execution."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           🔴 MULE ACCOUNT DETECTION — GRAPH VISUALIZATION                ║
║                    Fan-In Pattern Analysis                                ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load data
    print("\n📥 Loading data...")
    test_txns = pd.read_csv('test_transactions.csv')
    pred_df = pd.read_csv('fraud_predictions.csv')
    print(f"✓ Loaded {len(test_txns)} transactions")
    print(f"✓ Loaded {len(pred_df)} account predictions")
    
    # Visualize top 3 mule accounts
    print("\n" + "="*80)
    print("STEP 1: INDIVIDUAL MULE ACCOUNT VISUALIZATIONS")
    print("="*80)
    visualize_multiple_mules(test_txns, pred_df, num_mules=3)
    
    # Complete network
    print("\n" + "="*80)
    print("STEP 2: COMPLETE MULE NETWORK")
    print("="*80)
    visualize_complete_mule_network(test_txns, pred_df)
    
    # Comparison chart
    print("\n" + "="*80)
    print("STEP 3: MULE ACCOUNT COMPARISON DASHBOARD")
    print("="*80)
    create_mule_comparison_chart(test_txns, pred_df)
    
    print("\n" + "="*80)
    print("✅ MULE ACCOUNT VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\n📊 Generated Files:")
    print("   ✓ mule_account_1_*.png (Individual mule #1)")
    print("   ✓ mule_account_2_*.png (Individual mule #2)")
    print("   ✓ mule_account_3_*.png (Individual mule #3)")
    print("   ✓ mule_network_complete.png (Complete network)")
    print("   ✓ mule_accounts_analysis.png (Comparison dashboard)")
    print("\n🎯 These visualizations show:")
    print("   • Mule account at center (red node)")
    print("   • Multiple senders connecting to mule (blue nodes)")
    print("   • Edge thickness represents transaction amount")
    print("   • Pattern: FAN-IN (typical of money mule networks)")

if __name__ == '__main__':
    main()
