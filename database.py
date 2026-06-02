import sqlite3
import json
import os
import hashlib
import secrets
from datetime import datetime

DB_PATH = "rag_history.db"

# =====================================================
# 🔐 Password Hashing & Verification (PBKDF2-HMAC-SHA256)
# =====================================================

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hash_val = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    )
    return f"{salt}${hash_val.hex()}"

def verify_password(stored_hash: str, provided_password: str) -> bool:
    try:
        salt, hash_hex = stored_hash.split('$')
        hash_val = hashlib.pbkdf2_hmac(
            'sha256', 
            provided_password.encode('utf-8'), 
            salt.encode('utf-8'), 
            100000
        )
        return hash_val.hex() == hash_hex
    except Exception:
        return False

# =====================================================
# 📊 Database Initialization
# =====================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Chat Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            metrics TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            theme_preference TEXT NOT NULL DEFAULT 'dark',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    
    # Gracefully add theme_preference column to existing databases if it's missing
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN theme_preference TEXT NOT NULL DEFAULT 'dark'")
        conn.commit()
    except sqlite3.OperationalError:
        pass
        
    conn.close()


# =====================================================
# 👥 User Management Methods
# =====================================================

def create_user(username, email, password):
    email_lower = email.strip().lower()
    username_strip = username.strip()
    # Seed only harsh.verma@freseniusmedicalcare.com as admin
    role = "admin" if email_lower == "harsh.verma@freseniusmedicalcare.com" else "user"
    pwd_hash = hash_password(password)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, role, theme_preference)
            VALUES (?, ?, ?, ?, 'dark')
        """, (username_strip, email_lower, pwd_hash, role))
        conn.commit()
        user_id = cursor.lastrowid
        return True, user_id
    except sqlite3.IntegrityError:
        return False, "Username or Email already registered."
    finally:
        conn.close()

def authenticate_user(email, password):
    email_lower = email.strip().lower()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, email, password_hash, role, theme_preference FROM users
        WHERE email = ?
    """, (email_lower,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        user_id, username, db_email, stored_hash, role, theme_pref = row
        if verify_password(stored_hash, password):
            return {
                "id": user_id,
                "username": username,
                "email": db_email,
                "role": role,
                "theme_preference": theme_pref
            }
    return None

def get_user_by_credentials(user_id, email):
    email_lower = email.strip().lower()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, email, role, theme_preference FROM users
        WHERE id = ? AND email = ?
    """, (user_id, email_lower))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        uid, username, db_email, role, theme_pref = row
        return {
            "id": uid,
            "username": username,
            "email": db_email,
            "role": role,
            "theme_preference": theme_pref
        }
    return None

# =====================================================
# 💬 Chat Message History Methods
# =====================================================

def save_message(session_id, role, content, sources=None, metrics=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (session_id, role, content, sources, metrics)
        VALUES (?, ?, ?, ?, ?)
    """, (
        session_id, 
        role, 
        content, 
        json.dumps(sources) if sources else None, 
        json.dumps(metrics) if metrics else None
    ))
    conn.commit()
    conn.close()

def load_messages(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, sources, metrics FROM messages 
        WHERE session_id = ? 
        ORDER BY timestamp ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "role": row[0],
            "content": row[1],
            "sources": json.loads(row[2]) if row[2] else [],
            "metrics": json.loads(row[3]) if row[3] else {}
        })
    return history

def clear_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

# =====================================================
# 👥 User Administration Methods (Admin Only)
# =====================================================

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, role, created_at FROM users ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    users = []
    for r in rows:
        users.append({
            "id": r[0],
            "username": r[1],
            "email": r[2],
            "role": r[3],
            "created_at": r[4]
        })
    return users

def update_user(user_id, new_username, new_role):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE users 
            SET username = ?, role = ?
            WHERE id = ?
        """, (new_username.strip(), new_role, user_id))
        conn.commit()
        return True, "User updated successfully."
    except sqlite3.IntegrityError:
        return False, "Username already in use."
    finally:
        conn.close()

def delete_user_and_history(user_id, username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Delete history from messages table
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (username,))
        # Delete user record from users table
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return True
    except Exception as e:
        return False
    finally:
        conn.close()

def update_user_theme(user_id, theme_mode):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET theme_preference = ? WHERE id = ?", (theme_mode, user_id))
    conn.commit()
    conn.close()
