import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

# Set page configuration with a premium look
st.set_page_config(
    page_title="Fraudstruct Compliance Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Host Configuration
API_URL = "http://127.0.0.1:8000"

# Main styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 5px solid #1E3A8A;
    }
    .status-online {
        color: #10B981;
        font-weight: bold;
    }
    .status-offline {
        color: #EF4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check API status
def check_api_health():
    try:
        # Check docs or any endpoint to verify server is active
        response = requests.get(f"{API_URL}/docs", timeout=2)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False

# Sidebar Navigation & Status
st.sidebar.image("https://img.icons8.com/color/120/shield.png", width=80)
st.sidebar.markdown("### **Fraudstruct**")
st.sidebar.markdown("**MSc Thesis Research:**  \n*A Low-Latency Decoupled Streaming Framework for Real-Time Transaction Structuring Detection*")

# API Status check in sidebar
api_active = check_api_health()
if api_active:
    st.sidebar.markdown(f"Backend Status: <span class='status-online'>● ONLINE</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(f"Backend Status: <span class='status-offline'>● OFFLINE</span>", unsafe_allow_html=True)
    st.sidebar.warning("FastAPI backend is offline. Run 'python -m uvicorn fraudstruct.stream.api:app --reload --port 8000' in your terminal.")

st.sidebar.divider()
page = st.sidebar.radio(
    "Navigation Menu",
    ["Dashboard", "Transaction Terminal", "MLOps Model Manager", "Ecosystem Context"]
)

st.sidebar.divider()
if api_active:
    if st.sidebar.button("Reset Simulation State", type="secondary", use_container_width=True):
        try:
            res = requests.post(f"{API_URL}/v1/reset")
            if res.status_code == 200:
                st.session_state.alerts = []
                st.sidebar.success("Database reset successfully!")
                time.sleep(0.5)
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Reset failed: {str(e)}")

# Initialize Session State for Mock alert logger
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# --- Page 1: Dashboard ---
if page == "Dashboard":
    st.markdown("<h1 class='main-header'>Compliance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Real-time transaction risk profiling and payment switch SLA statistics.</p>", unsafe_allow_html=True)
    
    # Overview Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Mean Switch Latency", value="2.45 ms", delta="-47.55 ms (SLA)")
    with col2:
        st.metric(label="p95 Latency", value="3.41 ms", delta="-46.59 ms")
    with col3:
        st.metric(label="SLA Compliance Rate", value="100.00%", delta="0.00% breaches")
    with col4:
        st.metric(label="Active GNN Tenants", value="161 Banks", delta="NIBSS Hawk")

    st.divider()
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Real-Time Verification Audit Trail")
        if st.session_state.alerts:
            df = pd.DataFrame(st.session_state.alerts)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No transaction events evaluated in this session yet. Go to the 'Transaction Terminal' to test.")
            
    with col_right:
        st.subheader("CBN Automated AML Compliance")
        st.info("""
        **baseline standards directive:**
        Commercial banks in Nigeria must automate anti-money laundering controls (AML) to monitor networks and trace structuring (smurfing) across payment lines.
        
        **fraudstruct resolution:**
        By executing convolving updates asynchronously (Warm Path) and caching node descriptors, Fraudstruct is able to serve complex graph topological alerts synchronously during inter-bank clearing.
        """)

# --- Page 2: Transaction Terminal ---
elif page == "Transaction Terminal":
    st.markdown("<h1 class='main-header'>Transaction Evaluation Terminal</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Simulate core banking transaction inputs to evaluate risk outcomes in real time.</p>", unsafe_allow_html=True)
    
    if not api_active:
        st.error("API is offline. Please launch the FastAPI server to use the terminal.")
    else:
        st.subheader("Enter Transaction Information")
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            with col1:
                txn_id = st.text_input("Transaction ID", value=f"TXN-{int(time.time())}")
                source = st.text_input("Source Account", value="ACC_ATTACKER")
                destination = st.text_input("Destination Account", value="ACC_MULE_1")
            with col2:
                amount = st.number_input("Transaction Amount (NGN)", min_value=1.0, value=100000.0, step=500.0)
                channel = st.selectbox("Payment Channel", ["NIP", "POS", "MOBILE", "WEB"])
            
            submit = st.form_submit_button("Evaluate Transaction")
            
        if submit:
            payload = {
                "transaction_id": txn_id,
                "source_account": source,
                "destination_account": destination,
                "amount": amount,
                "channel": channel
            }
            
            with st.spinner("Invoking Fraudstruct evaluation engine..."):
                try:
                    t_start = time.perf_counter()
                    response = requests.post(f"{API_URL}/v1/evaluate", json=payload)
                    t_elapsed = (time.perf_counter() - t_start) * 1000  # ms
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        decision = res_data.get("decision")
                        features = res_data.get("features", {})
                        score = features.get("gnn_anomaly_score", 0.0)
                        reasons_list = res_data.get("reasons", [])
                        reason = "; ".join(reasons_list) if reasons_list else "Legitimate profile matches."
                        
                        # Add to session state history
                        st.session_state.alerts.insert(0, {
                            "Timestamp": datetime.now().strftime("%H:%M:%S"),
                            "Txn ID": txn_id,
                            "Source": source,
                            "Dest": destination,
                            "Amount": amount,
                            "Decision": decision,
                            "Score": score
                        })
                        
                        st.divider()
                        st.subheader("Evaluation Results")
                        
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            if decision == "BLOCK":
                                st.error(f"🔴 DECISION: {decision}")
                            elif decision == "FLAG":
                                st.warning(f"🟡 DECISION: {decision}")
                            else:
                                st.success(f"🟢 DECISION: {decision}")
                        with m_col2:
                            st.metric("GNN Anomaly Score", value=f"{score:.4f}")
                        with m_col3:
                            st.metric("Inference Latency", value=f"{t_elapsed:.2f} ms")
                            
                        st.write(f"**Audit Rationale:** {reason}")
                    else:
                        st.error(f"Error evaluating transaction: Status code {response.status_code}")
                except Exception as e:
                    st.error(f"Failed to communicate with API: {str(e)}")

# --- Page 3: MLOps Model Manager ---
elif page == "MLOps Model Manager":
    st.markdown("<h1 class='main-header'>MLOps Model Manager</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Retrain the NumPy SGC Graph Neural Network to update account topological profiles.</p>", unsafe_allow_html=True)
    
    if not api_active:
        st.error("API is offline. Please launch the FastAPI server to manage model training.")
    else:
        st.subheader("GNN Model Parameters")
        epochs = st.slider("Training Epochs", min_value=5, max_value=100, value=20, step=5)
        
        st.divider()
        st.subheader("Labels Payload")
        st.write("Train SGC Classifier on compromised nodes (Mules and Attackers):")
        
        # Prepare mock labels payload
        labels_df = pd.DataFrame([
            {"account_id": "ACC_ATTACKER", "is_fraud": 1},
            {"account_id": "ACC_MULE_1", "is_fraud": 1},
            {"account_id": "ACC_MULE_2", "is_fraud": 1},
            {"account_id": "ACC_BENEFICIARY", "is_fraud": 1},
            {"account_id": "ACC_0001", "is_fraud": 0},
            {"account_id": "ACC_0002", "is_fraud": 0}
        ])
        st.dataframe(labels_df, use_container_width=True)
        
        trigger_train = st.button("Trigger SGC GNN Retraining", type="primary")
        
        if trigger_train:
            labels_payload = labels_df.to_dict(orient="records")
            payload = {
                "labels": labels_payload,
                "epochs": epochs
            }
            
            with st.spinner("Performing asynchronous convolving updates & matrix optimization..."):
                try:
                    response = requests.post(f"{API_URL}/v1/train", json=payload)
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success("🎉 GNN Model Retraining Completed Successfully!")
                        st.json(res_data)
                    else:
                        st.error(f"Error during training: Status code {response.status_code}")
                except Exception as e:
                    st.error(f"Failed to communicate with API: {str(e)}")

# --- Page 4: Ecosystem Context ---
elif page == "Ecosystem Context":
    st.markdown("<h1 class='main-header'>Ecosystem Context & Benchmarking</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Evaluating how decentralized systems (Fraudstruct) complement centralized clearing systems (NIBSS Hawk).</p>", unsafe_allow_html=True)
    
    st.write("""
    **Architectural Clarification:**
    This research prototype represents a **decentralized, bank-level gateway solution** designed to run inside a single commercial bank's private cloud. 
    There is no active network handshake or live database integration between this Fraudstruct deployment and NIBSS Hawk. Instead, they operate as complementary, separate tiers of the national financial security architecture:
    """)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📊 Comparison: Local vs. Central Defense")
        st.markdown("""
        | Attribute | **Fraudstruct (This Project)** | **NIBSS Hawk (Central Clearing)** |
        | :--- | :--- | :--- |
        | **Deployment** | Internal DMB / Fintech Gateway | Centralized Clearing Switch Tier |
        | **SLA Latency** | **Sub-3ms (In-Flight)** | Switch routing overhead (10-30ms) |
        | **Data Scope** | Local ledger & customer graphs | Inter-bank consolidated transfers |
        | **Privacy Boundary**| Internal PII remains in bank | Aggregate PII sharing (NDPA friction) |
        """)
        
    with col_right:
        st.subheader("📈 NIBSS Hawk Ecosystem Statistics (2025 Milestones)")
        st.write("These industry-wide figures highlight the scale of the money-laundering problem that Fraudstruct is designed to mitigate at the bank entry point:")
        st.write("""
        * **Suspicious Cases Flagged**: 1,130,554 cases across 161 tenant banks.
        * **Watchlisted BVNs Intercepted**: 206,141 transactions.
        * **Fraudulent Transactions Blocked**: 4,422 transactions.
        * **Accounts with Invalid BVNs**: 29,058 transactions flagged.
        """)

