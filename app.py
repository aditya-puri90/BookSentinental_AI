import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from main import SmartLibraryMonitor, get_db_connection
from export_csv import export_to_csv

st.set_page_config(page_title="Smart Library Monitor", layout="wide")

st.title("📚 Smart Library Shelf Monitor")

def get_recent_displacements():
    try:
        conn = get_db_connection()
        query = "SELECT * FROM book_displacements ORDER BY displacement_time DESC LIMIT 10"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# Sidebar for configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
start_button = st.sidebar.button("Start Monitoring")
stop_button = st.sidebar.button("Stop Monitoring")
export_button = st.sidebar.button("Export to CSV now")

if export_button:
    msg = export_to_csv()
    st.sidebar.success(msg)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("Recent Displacements")
    log_placeholder = st.empty()

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    if start_button:
        monitor = SmartLibraryMonitor()
        st.toast("Monitoring Started!")
        
        while cap.isOpened():
            if stop_button:
                break
                
            ret, frame = cap.read()
            if not ret:
                st.info("End of video.")
                break
            
            # Resize frame for faster processing
            # Maintain aspect ratio
            height, width = frame.shape[:2]
            max_width = 800
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Process frame
            processed_frame = monitor.process_frame(frame)
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update logs occasionally (e.g., every 10 frames to save DB hits)
            if monitor.frame_count % 10 == 0:
                df = get_recent_displacements()
                if not df.empty:
                    log_placeholder.dataframe(df[['book_id', 'book_name', 'displacement_time']], hide_index=True)
                else:
                    log_placeholder.info("No displacements yet.")

        cap.release()
        os.remove(tfile.name)
        
        # Auto-export on stop
        msg = export_to_csv()
        st.toast(f"Monitoring Stopped. {msg}")
    
else:
    st.info("Please upload a video file to begin.")
