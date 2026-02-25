import sqlite3
import streamlit as st
from passlib.hash import bcrypt
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List, Tuple
import os

DB_PATH = "users.db"
DEFAULT_ADMIN_USERNAME = "anand"
DEFAULT_ADMIN_PASSWORD = "anand123"
DEFAULT_ADMIN_ROLE = "admin"

SESSION_TIMEOUT = 30 * 60  # seconds

class AuthManager:
    def __init__(self):
        self.db_path = DB_PATH
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize the SQLite database and create users table if it doesn't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        login_attempts INTEGER DEFAULT 0,
                        locked_until TIMESTAMP
                    )
                """)

                cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (DEFAULT_ADMIN_USERNAME,))
                admin_exists = cursor.fetchone()[0]
                
                if not admin_exists:
                    password_hash = bcrypt.hash(DEFAULT_ADMIN_PASSWORD)
                    cursor.execute("""
                        INSERT INTO users (username, password_hash, role, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (DEFAULT_ADMIN_USERNAME, password_hash, DEFAULT_ADMIN_ROLE, datetime.now()))
                    
                    conn.commit()
                    print(f"Default admin user '{DEFAULT_ADMIN_USERNAME}' created successfully")
                
                conn.commit()
                
        except Exception as e:
            st.error(f"Database initialization failed: {str(e)}")
            st.stop()
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hash(password)
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.verify(password, password_hash)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate a user and return user info if successful"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, username, password_hash, role, login_attempts, locked_until
                    FROM users WHERE username = ?
                """, (username,))
                
                user_data = cursor.fetchone()
                
                if not user_data:
                    return None
                
                user_id, db_username, password_hash, role, login_attempts, locked_until = user_data
                
                if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
                    return {"error": "Account is temporarily locked due to too many failed attempts"}
                
                if self.verify_password(password, password_hash):
                    cursor.execute("""
                        UPDATE users 
                        SET login_attempts = 0, locked_until = NULL, last_login = ?
                        WHERE id = ?
                    """, (datetime.now(), user_id))
                    
                    conn.commit()
                    
                    return {
                        "id": user_id,
                        "username": db_username,
                        "role": role,
                        "authenticated": True
                    }
                else:
                    new_attempts = login_attempts + 1
                    locked_until = None
                    
                    if new_attempts >= 5:
                        locked_until = datetime.now() + timedelta(minutes=15)
                    
                    cursor.execute("""
                        UPDATE users 
                        SET login_attempts = ?, locked_until = ?
                        WHERE id = ?
                    """, (new_attempts, locked_until, user_id))
                    
                    conn.commit()
                    
                    if new_attempts >= 5:
                        return {"error": "Account locked due to too many failed attempts. Try again in 15 minutes."}
                    else:
                        return {"error": f"Invalid password. {5 - new_attempts} attempts remaining."}
                        
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return None
    
    def create_user(self, username: str, password: str, role: str) -> Tuple[bool, str]:
        """Create a new user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
                if cursor.fetchone()[0] > 0:
                    return False, "Username already exists"
                
                password_hash = self.hash_password(password)
                cursor.execute("""
                    INSERT INTO users (username, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?)
                """, (username, password_hash, role, datetime.now()))
                
                conn.commit()
                return True, "User created successfully"
                
        except Exception as e:
            return False, f"Error creating user: {str(e)}"
    
    def delete_user(self, user_id: int) -> Tuple[bool, str]:
        """Delete a user by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
                user = cursor.fetchone()
                
                if not user:
                    return False, "User not found"
                
                if user[0] == DEFAULT_ADMIN_USERNAME:
                    return False, "Cannot delete the default admin user"
                
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                conn.commit()
                
                return True, f"User '{user[0]}' deleted successfully"
                
        except Exception as e:
            return False, f"Error deleting user: {str(e)}"
    
    def reset_user_password(self, user_id: int, new_password: str) -> Tuple[bool, str]:
        """Reset a user's password"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
                user = cursor.fetchone()
                
                if not user:
                    return False, "User not found"
                
                password_hash = self.hash_password(new_password)
                cursor.execute("""
                    UPDATE users 
                    SET password_hash = ?, login_attempts = 0, locked_until = NULL
                    WHERE id = ?
                """, (password_hash, user_id))
                
                conn.commit()
                return True, f"Password for user '{user[0]}' reset successfully"
                
        except Exception as e:
            return False, f"Error resetting password: {str(e)}"
    
    def get_all_users(self) -> List[Dict]:
        """Get all users (for admin view)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, username, role, created_at, last_login, login_attempts
                    FROM users ORDER BY created_at DESC
                """)
                
                users = []
                for row in cursor.fetchall():
                    users.append({
                        "id": row[0],
                        "username": row[1],
                        "role": row[2],
                        "created_at": row[3],
                        "last_login": row[4],
                        "login_attempts": row[5]
                    })
                
                return users
                
        except Exception as e:
            st.error(f"Error fetching users: {str(e)}")
            return []
    
    def update_user_role(self, user_id: int, new_role: str) -> Tuple[bool, str]:
        """Update a user's role"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
                user = cursor.fetchone()
                
                if not user:
                    return False, "User not found"
                
                if user[0] == DEFAULT_ADMIN_USERNAME:
                    return False, "Cannot change the default admin user's role"
                
                cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
                conn.commit()
                
                return True, f"Role for user '{user[0]}' updated to '{new_role}'"
                
        except Exception as e:
            return False, f"Error updating user role: {str(e)}"


def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None

def check_session_timeout():
    """Check if the current session has timed out"""
    if not st.session_state.authenticated or not st.session_state.login_time:
        return True
    
    elapsed_time = time.time() - st.session_state.login_time
    if elapsed_time > SESSION_TIMEOUT:
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.session_state.login_time = None
        return True
    
    st.session_state.login_time = time.time()
    return False

def login_user(user_info: Dict):
    """Set session state for logged in user"""
    st.session_state.authenticated = True
    st.session_state.user_info = user_info
    st.session_state.login_time = time.time()

def logout_user():
    """Clear session state for logged out user"""
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.login_time = None

def is_authenticated() -> bool:
    """Check if user is authenticated and session is valid"""
    init_session_state()
    return st.session_state.authenticated and not check_session_timeout()

def get_current_user() -> Optional[Dict]:
    """Get current user info if authenticated"""
    if is_authenticated():
        return st.session_state.user_info
    return None

def is_admin() -> bool:
    """Check if current user is an admin"""
    user = get_current_user()
    return user and user.get('role') == 'admin'

auth_manager = AuthManager() 