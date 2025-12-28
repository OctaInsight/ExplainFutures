"""
ExplainFutures - Supabase Database Manager (SIMPLIFIED VERSION)
Avoids proxy and advanced client options
"""

import streamlit as st
from supabase import create_client, Client
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime, timedelta
import json
import bcrypt


class SupabaseManager:
    """Simplified Supabase manager without advanced options"""
    
    def __init__(self):
        """Initialize Supabase client from Streamlit secrets"""
        try:
            # Get credentials
            self.url = st.secrets["supabase"]["url"]
            self.key = st.secrets["supabase"]["key"]
            
            # Simple client creation (no options)
            self.client = create_client(self.url, self.key)
            
            # Demo IDs
            self.demo_user_id = st.secrets["app"]["demo_user_id"]
            self.demo_project_id = st.secrets["app"]["demo_project_id"]
            
            # Test connection
            self.client.table('users').select('user_id').limit(1).execute()
            
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            st.info("💡 Try updating Supabase: pip install supabase==1.0.4")
            raise
    
    # ========================================================================
    # USER MANAGEMENT
    # ========================================================================
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        try:
            result = self.client.table('users').select('*').eq('username', username).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            st.error(f"Error fetching user: {str(e)}")
            return None
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verify user password"""
        user = self.get_user_by_username(username)
        if not user:
            return False
        
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'), 
                user['password_hash'].encode('utf-8')
            )
        except Exception:
            return False
    
    def login_user(self, username: str, password: str, 
                   ip_address: str = None, user_agent: str = None) -> Optional[Dict]:
        """Login user"""
        user = self.get_user_by_username(username)
        
        if not user or not user.get('is_active'):
            return None
        
        if not self.verify_password(username, password):
            # Update failed attempts
            try:
                self.client.table('users').update({
                    'failed_login_attempts': user.get('failed_login_attempts', 0) + 1,
                    'last_failed_login': datetime.now().isoformat()
                }).eq('user_id', user['user_id']).execute()
            except:
                pass
            return None
        
        # Update last login
        try:
            self.client.table('users').update({
                'last_login': datetime.now().isoformat(),
                'last_login_ip': ip_address,
                'last_login_user_agent': user_agent,
                'login_count': user.get('login_count', 0) + 1,
                'failed_login_attempts': 0
            }).eq('user_id', user['user_id']).execute()
        except:
            pass
        
        # Log login
        try:
            self.client.table('user_login_history').insert({
                'user_id': user['user_id'],
                'login_successful': True,
                'ip_address': ip_address,
                'user_agent': user_agent
            }).execute()
        except:
            pass
        
        return user
    
    def request_password_reset(self, email: str) -> Dict[str, Any]:
        """Request password reset"""
        try:
            # Check if email exists
            result = self.client.table('users').select('*').eq('email', email).execute()
            
            if not result.data:
                return {
                    'success': False,
                    'message': f"This email is not registered. Please request a subscription at sales@octainsight.com",
                    'email_sent': False
                }
            
            user = result.data[0]
            
            # Try to send reset email via Supabase auth
            try:
                # This uses Supabase's built-in password reset
                self.client.auth.reset_password_for_email(email)
                
                return {
                    'success': True,
                    'message': f"Password reset instructions sent to {email}. Check your inbox.",
                    'email_sent': True
                }
            except:
                # Fallback if email fails
                return {
                    'success': True,
                    'message': f"Password reset requested. Please contact sales@octainsight.com for assistance.",
                    'email_sent': False
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"An error occurred. Please contact sales@octainsight.com",
                'error': str(e)
            }
    
    def is_demo_user(self, user_id: str) -> bool:
        """Check if user is demo user"""
        return str(user_id) == str(self.demo_user_id)
    
    # ========================================================================
    # DEMO SESSION MANAGEMENT
    # ========================================================================
    
    def create_demo_session(self, user_id: str, project_id: str, 
                           duration_seconds: int = 1800) -> Optional[Dict]:
        """Create demo session"""
        try:
            session_token = str(uuid.uuid4())
            expires_at = datetime.now() + timedelta(seconds=duration_seconds)
            
            result = self.client.table('demo_sessions').insert({
                'user_id': user_id,
                'project_id': project_id,
                'session_token': session_token,
                'expires_at': expires_at.isoformat(),
                'cleanup_required': True
            }).execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            st.error(f"Failed to create demo session: {str(e)}")
            return None
    
    def end_demo_session(self, session_id: str):
        """End demo session and cleanup"""
        try:
            # Call cleanup function
            self.client.rpc('cleanup_demo_session', {'p_session_id': session_id}).execute()
        except Exception as e:
            st.warning(f"Demo cleanup warning: {str(e)}")
    
    # ========================================================================
    # PROJECT MANAGEMENT
    # ========================================================================
    
    def get_user_projects(self, user_id: str, include_collaborations: bool = True) -> List[Dict]:
        """Get user's projects"""
        try:
            query = self.client.table('projects').select('*').eq('owner_id', user_id)
            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            st.error(f"Error fetching projects: {str(e)}")
            return []
    
    def update_project_progress(self, project_id: str, workflow_state: str = None,
                               current_page: int = None, completion_percentage: int = None):
        """Update project progress"""
        try:
            update_data = {'updated_at': datetime.now().isoformat()}
            
            if workflow_state:
                update_data['workflow_state'] = workflow_state
            if current_page:
                update_data['current_page'] = current_page
            if completion_percentage is not None:
                update_data['completion_percentage'] = completion_percentage
            
            self.client.table('projects').update(update_data).eq('project_id', project_id).execute()
        except Exception as e:
            st.warning(f"Progress update warning: {str(e)}")


# Singleton pattern with caching
@st.cache_resource
def get_db_manager() -> SupabaseManager:
    """Get cached database manager instance"""
    return SupabaseManager()
