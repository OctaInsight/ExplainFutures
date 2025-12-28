"""
Database Diagnostic Script - Save as pages/99_Diagnostic.py
Run this to check what's in your Supabase database
"""

import streamlit as st

st.set_page_config(page_title="Database Diagnostic", page_icon="🔍")

st.title("🔍 Database Diagnostic")

try:
    from core.database.supabase_manager import get_db_manager
    
    st.success("✅ Database manager imported successfully")
    
    db = get_db_manager()
    st.success("✅ Database connection established")
    
    # Check users table
    st.markdown("---")
    st.markdown("## 👥 Users in Database")
    
    try:
        # Get ALL users
        result = db.client.table('users').select('username, email, is_active, is_demo_user, password_hash').execute()
        
        if result.data:
            st.success(f"✅ Found {len(result.data)} users")
            
            import pandas as pd
            df = pd.DataFrame(result.data)
            df['hash_exists'] = df['password_hash'].apply(lambda x: bool(x) and len(x) > 0)
            df['hash_length'] = df['password_hash'].apply(lambda x: len(x) if x else 0)
            st.dataframe(df[['username', 'email', 'is_active', 'is_demo_user', 'hash_exists', 'hash_length']])
            
            for user in result.data:
                with st.expander(f"User: {user.get('username', 'NO USERNAME')}"):
                    # Don't show password hash
                    display_user = {k: v for k, v in user.items() if k != 'password_hash'}
                    display_user['password_hash'] = f"[{len(user.get('password_hash', ''))} chars]"
                    st.json(display_user)
                    
                    # Test password
                    st.markdown("### Password Test")
                    
                    if user.get('username') == 'demo':
                        st.write("Testing password: demo123")
                        try:
                            matches = db.verify_password('demo', 'demo123')
                            if matches:
                                st.success("✅ Password 'demo123' matches!")
                            else:
                                st.error("❌ Password 'demo123' does NOT match!")
                        except Exception as e:
                            st.error(f"Password verification error: {str(e)}")
                    
                    elif user.get('username') == 'octainsight':
                        st.write("Testing password: OctaTest2024!")
                        try:
                            matches = db.verify_password('octainsight', 'OctaTest2024!')
                            if matches:
                                st.success("✅ Password 'OctaTest2024!' matches!")
                            else:
                                st.error("❌ Password 'OctaTest2024!' does NOT match!")
                        except Exception as e:
                            st.error(f"Password verification error: {str(e)}")
        else:
            st.error("❌ NO USERS FOUND IN DATABASE!")
            st.warning("You need to run CREATE_DEMO_AND_TEST_ACCOUNTS.sql")
            
    except Exception as e:
        st.error(f"❌ Error querying users table: {str(e)}")
        st.exception(e)
    
    # Test login function
    st.markdown("---")
    st.markdown("## 🔐 Test Login Function")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Test Demo Login")
        if st.button("Test: demo / demo123"):
            try:
                result = db.login_user('demo', 'demo123')
                if result:
                    st.success("✅ Login successful!")
                    st.json({k: v for k, v in result.items() if k != 'password_hash'})
                else:
                    st.error("❌ Login failed!")
                    
                    # Debug
                    user = db.get_user_by_username('demo')
                    if user:
                        st.warning("User exists in DB")
                        st.write(f"is_active: {user.get('is_active')}")
                        st.write(f"is_demo_user: {user.get('is_demo_user')}")
                    else:
                        st.error("User NOT found in DB")
            except Exception as e:
                st.error(f"Login error: {str(e)}")
                st.exception(e)
    
    with col2:
        st.markdown("### Test Octainsight Login")
        if st.button("Test: octainsight / OctaTest2024!"):
            try:
                result = db.login_user('octainsight', 'OctaTest2024!')
                if result:
                    st.success("✅ Login successful!")
                    st.json({k: v for k, v in result.items() if k != 'password_hash'})
                else:
                    st.error("❌ Login failed!")
                    
                    # Debug
                    user = db.get_user_by_username('octainsight')
                    if user:
                        st.warning("User exists in DB")
                        st.write(f"is_active: {user.get('is_active')}")
                    else:
                        st.error("User NOT found in DB")
            except Exception as e:
                st.error(f"Login error: {str(e)}")
                st.exception(e)

except ImportError as e:
    st.error(f"❌ Cannot import database manager: {str(e)}")
    st.exception(e)
except Exception as e:
    st.error(f"❌ Unexpected error: {str(e)}")
    st.exception(e)


# Quick SQL queries to run
st.markdown("---")
st.markdown("## 📝 SQL Queries to Run in Supabase")

st.code("""
-- Check all users
SELECT username, email, is_active, is_demo_user, 
       LENGTH(password_hash) as hash_length,
       created_at 
FROM users;

-- Verify demo user
SELECT username, email, is_active, 
       SUBSTRING(password_hash, 1, 20) as hash_preview
FROM users WHERE username = 'demo';
""", language="sql")
