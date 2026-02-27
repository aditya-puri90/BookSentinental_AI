import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from main import SmartLibraryMonitor, get_db_connection
from export_csv import export_to_csv

# Must be the first Streamlit command
st.set_page_config(
    page_title="Smart Library Monitor", 
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
        padding-top: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 25px;
        font-weight: 500;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1E3A8A;
    }
    
    /* Info text styling */
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
    
    /* Adjust main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Style cards/containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

def get_recent_displacements():
    try:
        conn = get_db_connection()
        query = "SELECT book_id as 'Book ID', book_name as 'Book Title', displacement_time as 'Time Recorded' FROM book_displacements ORDER BY displacement_time DESC LIMIT 15"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

def get_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM book_displacements")
        total_displacements = cur.fetchone()[0]
        
        # Get unique books displaced
        cur.execute("SELECT COUNT(DISTINCT book_id) FROM book_displacements")
        unique_books = cur.fetchone()[0]
        
        conn.close()
        return total_displacements, unique_books
    except:
        return 0, 0

# Header Section
st.markdown('<p class="main-header">📚 Smart Library Shelf Monitor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Computer Vision Tracking for Inventory Management</p>', unsafe_allow_html=True)

st.divider()

# Sidebar Configuration Console
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3389/3389081.png", width=70)
    st.title("Control Center")
    st.markdown("Upload a security camera feed to automatically track library shelf activity.")
    
    uploaded_file = st.file_uploader("📥 Upload Video Feed", type=['mp4', 'avi', 'mov'], help="Supported formats: MP4, AVI, MOV")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🕹️ System Controls")
    col_start, col_stop = st.columns(2)
    with col_start:
        start_button = st.button("▶️ Start Tracking", use_container_width=True, type="primary")
    with col_stop:
        stop_button = st.button("⏹️ Stop Feed", use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💾 Data Management")
    export_button = st.button("📊 Download CSV Report", use_container_width=True)
    
    if export_button:
        with st.spinner("Compiling data logs..."):
            msg = export_to_csv()
            st.success(f"✅ {msg}")
            
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ How it works", icon="🎓"):
        st.info("The system utilizes a YOLOv8 object detection model combined with EasyOCR to locate books and read their text spines. It constantly monitors for missing objects and records anomalies to the local SQLite database.")

# Main Layout with Metrics
metrics_placeholder = st.container()

# Two-column layout for Video Feed and Logs
col_feed, col_logs = st.columns([2.5, 1.5], gap="large")

with col_feed:
    st.markdown("### 📷 Live Surveillance Feed")
    feed_card = st.container(border=True)
    with feed_card:
        video_placeholder = st.empty()
        st.caption("Monitoring active shelf regions in real-time...")

with col_logs:
    st.markdown("### 📋 Activity Logs")
    log_card = st.container(border=True)
    with log_card:
        log_placeholder = st.empty()

# Setup default metrics
total_disp, unique_books = get_stats()
with metrics_placeholder:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="System Status", value="Idle ⚪")
    with m2:
        disp_metric = st.empty()
        disp_metric.metric(label="Total Shelf Interactions", value=total_disp, delta=None)
    with m3:
        unique_metric = st.empty()
        unique_metric.metric(label="Unique Books Moved", value=unique_books, delta=None)

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    if start_button:
        # Initialize Core Logic
        monitor = SmartLibraryMonitor()
        st.toast("🚀 Surveillance Session Started!", icon="🟢")
        
        # Update metrics to Active
        with m1:
            st.metric(label="System Status", value="Active 🟢", delta="Recording")
            
        frame_counter = 0
        
        # Determine approx total frames (some videos don't report accurately, so fallback provided)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1000 
        
        progress_bar = st.progress(0, text="Initializing video stream processing...")
        
        while cap.isOpened():
            if stop_button:
                st.toast("Monitoring manually terminated.", icon="🛑")
                break
                
            ret, frame = cap.read()
            if not ret:
                st.balloons()
                st.toast("End of video feed reached.", icon="🏁")
                break
            
            frame_counter += 1
            progress_val = min(frame_counter / total_frames, 1.0)
            progress_bar.progress(progress_val, text=f"Processing frame {frame_counter}...")
            
            # Efficient resizing
            height, width = frame.shape[:2]
            max_width = 800
            if width > max_width:
                scale = max_width / width
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

            # CV Processing Step
            processed_frame = monitor.process_frame(frame)
            
            # Color conversion for UI
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Render frame
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # DB & Log Checks
            if monitor.frame_count % 15 == 0:
                df = get_recent_displacements()
                if not df.empty:
                    log_placeholder.dataframe(df, hide_index=True, use_container_width=True)
                    t_disp, u_books = get_stats()
                    disp_metric.metric(label="Total Shelf Interactions", value=t_disp)
                    unique_metric.metric(label="Unique Books Moved", value=u_books)
                else:
                    log_placeholder.info("⏳ Scanning for anomalies...")

        # Cleanup process
        cap.release()
        try:
            os.remove(tfile.name)
        except:
            pass
        
        progress_bar.empty()
        
        # Final Export
        with st.spinner("Generating final audit report..."):
            msg = export_to_csv()
            st.toast(f"Session Logged: {msg}", icon="💾")
    else:
        # Idle state with video loaded
        video_placeholder.info("Video feed loaded. Press 'Start Tracking' in the control panel to begin analysis.")
        
        # Show past data
        df = get_recent_displacements()
        if not df.empty:
            log_placeholder.dataframe(df, hide_index=True, use_container_width=True)
        else:
            log_placeholder.info("No recorded history found in database.")
else:
    # No video uploaded
    with col_feed:
        video_placeholder.info("👋 Welcome! Please upload a security feed via the left panel to begin monitoring.")
    with col_logs:
        df = get_recent_displacements()
        if not df.empty:
            log_placeholder.dataframe(df, hide_index=True, use_container_width=True)
        else:
            log_placeholder.markdown("*Awaiting first recording...*")
