import streamlit as st
from auth_utils import auth_manager, login_user, logout_user, is_admin, get_current_user
import pandas as pd
from datetime import datetime

def render_login_page():
    """Render the login page"""
    st.set_page_config(
        page_title="Neura-IQ Multimodal AI Research Assistant - Login",
        page_icon="🔐",
        layout="centered"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1>🚀 Neura-IQ</h1>
                <p style='color: gray; font-size: 1.1em;'>Multimodal AI Research Assistant</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.form("login_form"):
            st.markdown("### 🔐 Sign In")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col1, col2 = st.columns(2)
            with col1:
                submit_button = st.form_submit_button("Login", type="primary", use_container_width=True)
            with col2:
                if st.form_submit_button("Clear", use_container_width=True):
                    st.rerun()
            
            if submit_button:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    result = auth_manager.authenticate_user(username, password)
                    
                    if result and result.get('authenticated'):
                        login_user(result)
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    elif result and result.get('error'):
                        st.error(result['error'])
                    else:
                        st.error("Invalid username or password")
        

def render_user_management():
    """Render the user management interface for admins"""
    st.header("👥 User Management")
    st.markdown("Manage users and their permissions")
    
    tab1, tab2, tab3 = st.tabs(["📋 View Users", "➕ Create User", "⚙️ Manage Users"])
    
    with tab1:
        render_view_users_tab()
    
    with tab2:
        render_create_user_tab()
    
    with tab3:
        render_manage_users_tab()

def render_view_users_tab():
    """Render the view users tab"""
    st.subheader("📋 All Users")
    
    users = auth_manager.get_all_users()
    
    if not users:
        st.info("No users found in the system.")
        return
    
    df_data = []
    for user in users:
        df_data.append({
            "ID": user['id'],
            "Username": user['username'],
            "Role": user['role'],
            "Created": user['created_at'][:19] if user['created_at'] else "N/A",
            "Last Login": user['last_login'][:19] if user['last_login'] else "Never",
            "Login Attempts": user['login_attempts']
        })
    
    df = pd.DataFrame(df_data)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(users))
    with col2:
        admin_count = sum(1 for user in users if user['role'] == 'admin')
        st.metric("Admins", admin_count)
    with col3:
        user_count = sum(1 for user in users if user['role'] == 'user')
        st.metric("Regular Users", user_count)
    with col4:
        active_users = sum(1 for user in users if user['last_login'])
        st.metric("Active Users", active_users)
    
    st.markdown("---")
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Username": st.column_config.TextColumn("Username", width="medium"),
            "Role": st.column_config.SelectboxColumn(
                "Role",
                options=["admin", "user"],
                width="small"
            ),
            "Created": st.column_config.TextColumn("Created", width="medium"),
            "Last Login": st.column_config.TextColumn("Last Login", width="medium"),
            "Login Attempts": st.column_config.NumberColumn("Login Attempts", width="small")
        }
    )

def render_create_user_tab():
    """Render the create user tab"""
    st.subheader("➕ Create New User")
    
    with st.form("create_user_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Username", placeholder="Enter username")
            new_password = st.text_input("Password", type="password", placeholder="Enter password")
        
        with col2:
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password")
            new_role = st.selectbox("Role", ["user", "admin"], index=0)
        
        st.markdown("**Password Requirements:**")
        st.markdown("- At least 6 characters long")
        st.markdown("- Should not be empty")
        
        submitted = st.form_submit_button("Create User", type="primary")
        
        if submitted:
            if not new_username or not new_password:
                st.error("Username and password are required")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                success, message = auth_manager.create_user(new_username, new_password, new_role)
                
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

def render_manage_users_tab():
    """Render the manage users tab"""
    st.subheader("⚙️ Manage Existing Users")
    
    users = auth_manager.get_all_users()
    
    if not users:
        st.info("No users found to manage.")
        return
    
    # User selection
    user_options = {f"{user['username']} ({user['role']})": user['id'] for user in users}
    selected_user_label = st.selectbox("Select User to Manage", list(user_options.keys()))
    
    if selected_user_label:
        selected_user_id = user_options[selected_user_label]
        selected_user = next((u for u in users if u['id'] == selected_user_id), None)
        
        if selected_user:
            st.markdown(f"**Managing:** {selected_user['username']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Change Role**")
                current_role = selected_user['role']
                new_role = st.selectbox(
                    "New Role",
                    ["admin", "user"],
                    index=0 if current_role == "admin" else 1,
                    key="role_select"
                )
                
                if st.button("Update Role", key="update_role"):
                    if new_role != current_role:
                        success, message = auth_manager.update_user_role(selected_user_id, new_role)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.info("Role is already set to this value")
            
            with col2:
                st.markdown("**Reset Password**")
                new_password = st.text_input("New Password", type="password", key="reset_pwd")
                confirm_new_password = st.text_input("Confirm New Password", type="password", key="confirm_reset_pwd")
                
                if st.button("Reset Password", key="reset_password"):
                    if not new_password:
                        st.error("Please enter a new password")
                    elif new_password != confirm_new_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        success, message = auth_manager.reset_user_password(selected_user_id, new_password)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

            st.markdown("---")
            st.markdown("**⚠️ Delete User**")
            
            if selected_user['username'] == 'anand':
                st.warning("Cannot delete the default admin user")
            else:
                if st.button("🗑️ Delete User", type="secondary"):
                    success, message = auth_manager.delete_user(selected_user_id)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

def render_sidebar_user_info():
    """Render user information in the sidebar"""
    user = get_current_user()
    
    if user:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 User Info")
        
        st.sidebar.markdown(f"**Username:** {user['username']}")
        st.sidebar.markdown(f"**Role:** {user['role'].title()}")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
        
        if is_admin():
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔧 Admin Panel")
            
            if st.sidebar.button("👥 User Management", use_container_width=True):
                st.session_state.current_mode = "user_management"
                st.rerun()

def render_back_to_main_button():
    """Render a back to main menu button"""
    if st.button("← Back to Main Menu", key="back_to_main"):
        st.session_state.current_mode = "main"
        st.rerun() 