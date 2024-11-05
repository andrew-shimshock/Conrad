import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from typing import Optional

class LogsInterface:
    def __init__(self, log_manager):
        self.log_manager = log_manager
        self.log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    def render_logs_interface(self):
        """Render the logs management interface."""
        st.header("System Logs")

        # Create tabs for different log views
        tabs = st.tabs(["Live Logs", "Log Analysis", "Log Files"])

        with tabs[0]:
            self._render_live_logs()

        with tabs[1]:
            self._render_log_analysis()

        with tabs[2]:
            self._render_log_files()

    def _render_live_logs(self):
        """Render real-time logs view."""
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_level = st.selectbox(
                "Filter by Level",
                ['All'] + self.log_levels
            )

        with col2:
            search_term = st.text_input("Search Logs", "")

        with col3:
            limit = st.number_input("Limit", min_value=10, 
                                  max_value=1000, value=100)

        # Get filtered logs
        logs = self.log_manager.get_recent_logs(
            level=None if selected_level == 'All' else selected_level,
            limit=limit,
            search_term=search_term
        )

        # Display logs in a dataframe
        if logs:
            df = pd.DataFrame(logs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Style the dataframe
            def color_level(val):
                colors = {
                    'DEBUG': 'lightgrey',
                    'INFO': 'lightgreen',
                    'WARNING': 'yellow',
                    'ERROR': 'lightcoral',
                    'CRITICAL': 'red'
                }
                return f'background-color: {colors.get(val, "white")}'

            styled_df = df.style.applymap(
                color_level, 
                subset=['level']
            )

            st.dataframe(styled_df, height=400)
        else:
            st.info("No logs found matching the criteria.")

    def _render_log_analysis(self):
        """Render log analysis and visualization."""
        # Time range selector
        time_ranges = {
            'Last Hour': timedelta(hours=1),
            'Last 24 Hours': timedelta(days=1),
            'Last 7 Days': timedelta(days=7)
        }
        selected_range = st.selectbox("Time Range", list(time_ranges.keys()))
        
        logs =
