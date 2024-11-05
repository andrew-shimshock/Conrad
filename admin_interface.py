import streamlit as st
from typing import Dict, List
from config_manager import ConfigManager
from log_manager import LogManager
import pandas as pd
from datetime import datetime

class AdminInterface:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.available_models = ["chatgpt", "claude", "local"]
        self.log_manager = LogManager()

    def render_login(self) -> bool:
        """Render admin login interface."""
        st.sidebar.header("Admin Login")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if self.config_manager.verify_admin_password(password):
                st.session_state.admin_authenticated = True
                return True
            else:
                st.sidebar.error("Invalid password")
        return False

    def render_admin_interface(self):
        """Render main admin interface."""
        st.header("Admin Dashboard")
        
        # Create tabs for different admin sections
        tab1, tab2, tab3 = st.tabs([
            "Routing Configuration", 
            "System Logs", 
            "System Status"
        ])
        
        with tab1:
            self._render_routing_config()
        with tab2:
            self._render_logs_interface()
        with tab3:
            self._render_system_status()

    def _render_routing_config(self):
        """Render routing configuration section."""
        st.header("Routing Configuration Management")
        
        # Category Management
        with st.expander("Add/Edit Category", expanded=True):
            self._render_category_management()

        # Existing Categories
        with st.expander("Existing Categories", expanded=True):
            self._render_existing_categories()

    def _render_category_management(self):
        """Render category management interface."""
        col1, col2 = st.columns(2)
        
        with col1:
            category_name = st.text_input("Category Name")
            keywords = st.text_area("Keywords (comma-separated)")
            model = st.selectbox("Model", self.available_models)
        
        with col2:
            prompt_template = st.text_area(
                "Prompt Template",
                height=200,
                help="Use {input} as placeholder for the user's question"
            )

        if st.button("Save Category"):
            if category_name and keywords and prompt_template:
                keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
                self.config_manager.add_category(
                    category_name,
                    keyword_list,
                    model,
                    prompt_template
                )
                st.success(f"Category '{category_name}' saved successfully!")
            else:
                st.error("Please fill in all fields")

    def _render_existing_categories(self):
        """Render existing categories interface."""
        categories = self.config_manager.get_categories()
        
        for category, config in categories.items():
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.subheader(category)
                    st.write(f"Model: {config['model']}")
                
                with col2:
                    st.write("Keywords:")
                    st.write(", ".join(config['keywords']))
                
                with col3:
                    if st.button("Delete", key=f"del_{category}"):
                        self.config_manager.remove_category(category)
                        st.experimental_rerun()
                
                st.markdown("---")

    def _render_logs_interface(self):
        """Render the logs visualization interface."""
        st.header("System Logs")
        
        # Log filtering options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hours = st.selectbox(
                "Time Range",
                [1, 6, 12, 24, 48, 72],
                index=3,
                help="Show logs from the last X hours"
            )
        
        with col2:
            log_level = st.selectbox(
                "Log Level",
                ["All", "INFO", "WARNING", "ERROR"],
                index=0
            )
            
        with col3:
            if st.button("Clear Logs"):
                if self.log_manager.clear_logs():
                    st.success("Logs cleared successfully")
                else:
                    st.error("Error clearing logs")
                    
        # Get and display logs
        logs = self.log_manager.get_logs(
            hours=hours,
            log_level=None if log_level == "All" else log_level
        )
        
        if logs:
            # Convert logs to DataFrame for better display
            df = pd.DataFrame(logs)
            
            # Style the DataFrame
            def color_level(val):
                colors = {
                    'ERROR': 'color: red',
                    'WARNING': 'color: orange',
                    'INFO': 'color: green'
                }
                return colors.get(val, '')
            
            styled_df = df.style.applymap(color_level, subset=['level'])
            
            # Display logs in a scrollable container
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400
            )
            
            # Download logs option
            if st.download_button(
                "Download Logs",
                df.to_csv(index=False).encode('utf-8'),
                "logs.csv",
                "text/csv",
                key='download-logs'
            ):
                st.success("Logs downloaded successfully!")
        else:
            st.info("No logs found for the selected criteria")

    def _render_system_status(self):
        """Render system status information."""
        st.header("System Status")
        
        # Display basic system statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Total Categories", 
                len(self.config_manager.get_categories())
            )
            
        with col2:
            try:
                logs = self.log_manager.get_logs(hours=24)
                error_count = sum(1 for log in logs if log['level'] == 'ERROR')
                st.metric("Errors (24h)", error_count)
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")
