"""
Add this to your diagnostic page or run as separate test
Save as: pages/98_Login_Test.py
"""

import streamlit as st

st.title("🔐 Detailed Login Test")

try:
    from core.database.supabase_manager import get_db_manager
    import bcrypt
    
    db = get_db_manager()
    st.success("✅ Database connected")
    
    # Test 1: Check if user exists
    st.markdown("## Test 1: Get User by Username")
    demo_user = db.get_user_by_username('demo')
    
    if demo_user:
        st.success("✅ Demo user found!")
        
        # Show user details (hide password)
        user_display = {k: v for k, v in demo_user.items() if k != 'password_hash'}
        user_display['password_hash_length'] = len(demo_user.get('password_hash', ''))
        st.json(user_display)
        
        # Check critical fields
        st.markdown("### Critical Field Checks:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            is_active = demo_user.get('is_active')
            if is_active is None:
                st.error("❌ is_active is NULL!")
            elif is_active == True:
                st.success(f"✅ is_active = {is_active}")
            else:
                st.error(f"❌ is_active = {is_active}")
        
        with col2:
            has_hash = bool(demo_user.get('password_hash'))
            if has_hash:
                st.success("✅ password_hash exists")
            else:
                st.error("❌ password_hash is missing!")
        
        # Test 2: Password verification
        st.markdown("## Test 2: Password Verification")
        
        password_to_test = 'demo123'
        st.write(f"Testing password: `{password_to_test}`")
        
        try:
            # Manual bcrypt check
            password_hash = demo_user.get('password_hash', '')
            if password_hash:
                matches = bcrypt.checkpw(
                    password_to_test.encode('utf-8'),
                    password_hash.encode('utf-8')
                )
                
                if matches:
                    st.success("✅ Password matches (manual bcrypt check)")
                else:
                    st.error("❌ Password does NOT match (manual bcrypt check)")
                    st.write(f"Hash format: {password_hash[:20]}...")
            else:
                st.error("❌ No password hash to check")
                
            # Using db.verify_password
            st.markdown("### Using db.verify_password():")
            verify_result = db.verify_password('demo', password_to_test)
            if verify_result:
                st.success("✅ db.verify_password() = True")
            else:
                st.error("❌ db.verify_password() = False")
                
        except Exception as e:
            st.error(f"Password check error: {str(e)}")
            st.exception(e)
        
        # Test 3: Full login flow
        st.markdown("## Test 3: Full Login Flow")
        
        if st.button("🧪 Test db.login_user('demo', 'demo123')"):
            try:
                result = db.login_user('demo', 'demo123', 'test_ip', 'test_agent')
                
                if result:
                    st.success("✅ LOGIN SUCCESSFUL!")
                    st.json({k: v for k, v in result.items() if k != 'password_hash'})
                else:
                    st.error("❌ LOGIN FAILED!")
                    
                    # Debug why it failed
                    st.markdown("### Debug Info:")
                    
                    # Re-check user
                    user_check = db.get_user_by_username('demo')
                    if not user_check:
                        st.error("❌ User not found in second check!")
                    elif not user_check.get('is_active'):
                        st.error(f"❌ User is not active: is_active = {user_check.get('is_active')}")
                    elif not db.verify_password('demo', 'demo123'):
                        st.error("❌ Password verification failed")
                    else:
                        st.error("❌ Unknown reason for failure")
                        
            except Exception as e:
                st.error(f"Login error: {str(e)}")
                st.exception(e)
    
    else:
        st.error("❌ Demo user NOT found!")
        
        # Check what users exist
        st.markdown("### Users in database:")
        result = db.client.table('users').select('username, email, is_active').execute()
        if result.data:
            st.dataframe(result.data)
        else:
            st.error("NO USERS IN DATABASE")

    # Test octainsight too
    st.markdown("---")
    st.markdown("## Test Octainsight User")
    
    if st.button("🧪 Test octainsight login"):
        octa_user = db.get_user_by_username('octainsight')
        if octa_user:
            st.success("✅ User found")
            st.write(f"is_active: {octa_user.get('is_active')}")
            
            # Test password
            matches = db.verify_password('octainsight', 'OctaTest2024!')
            if matches:
                st.success("✅ Password matches")
            else:
                st.error("❌ Password does NOT match")
            
            # Test full login
            result = db.login_user('octainsight', 'OctaTest2024!')
            if result:
                st.success("✅ Login successful!")
            else:
                st.error("❌ Login failed!")
        else:
            st.error("❌ User not found")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)


# SQL queries to fix common issues
st.markdown("---")
st.markdown("## 🔧 SQL Fixes (if needed)")

st.code("""
-- If is_active is NULL, set it to true:
UPDATE users SET is_active = true WHERE username = 'demo';
UPDATE users SET is_active = true WHERE username = 'octainsight';

-- If is_demo_user is NULL:
UPDATE users SET is_demo_user = true WHERE username = 'demo';
UPDATE users SET is_demo_user = false WHERE username = 'octainsight';

-- Verify:
SELECT username, is_active, is_demo_user, 
       LENGTH(password_hash) as hash_len
FROM users 
WHERE username IN ('demo', 'octainsight');
""", language="sql")
