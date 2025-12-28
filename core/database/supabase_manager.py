"""
ExplainFutures - Supabase Database Manager (PLAINTEXT PASSWORD VERSION)
⚠️ WARNING: This version uses plaintext passwords - NOT RECOMMENDED for production!
"""

import streamlit as st
from supabase import create_client, Client
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime, timedelta
import json


class SupabaseManager:
    """Simplified Supabase manager with plaintext password authentication"""
    
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
        """
        Verify user password (plaintext comparison)
        ⚠️ WARNING: This is insecure - passwords are stored in plaintext!
        """
        user = self.get_user_by_username(username)
        if not user:
            return False
        
        try:
            # Simple plaintext comparison
            return user.get('password') == password
        except Exception:
            return False
    
    def login_user(self, username: str, password: str, 
                   ip_address: str = None, user_agent: str = None) -> Optional[Dict]:
        """Login user with plaintext password verification"""
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
    
    def create_user(self, username: str, email: str, password: str, 
                    full_name: str = None, subscription_tier: str = 'free') -> Optional[Dict]:
        """
        Create new user with plaintext password
        ⚠️ WARNING: Password stored in plaintext - very insecure!
        """
        try:
            result = self.client.table('users').insert({
                'username': username,
                'email': email,
                'password': password,  # Plaintext password (INSECURE!)
                'full_name': full_name or username,
                'subscription_tier': subscription_tier,
                'is_active': True
            }).execute()
            
            return result.data[0] if result.data else None
            
        except Exception as e:
            st.error(f"Error creating user: {str(e)}")
            return None
    
    def update_password(self, user_id: str, new_password: str) -> bool:
        """
        Update user password
        ⚠️ WARNING: Password stored in plaintext - very insecure!
        """
        try:
            self.client.table('users').update({
                'password': new_password,  # Plaintext password (INSECURE!)
                'updated_at': datetime.now().isoformat()
            }).eq('user_id', user_id).execute()
            return True
        except Exception as e:
            st.error(f"Error updating password: {str(e)}")
            return False
    
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
            
            # Since we're using plaintext passwords, we could theoretically
            # send the password in email (EXTREMELY INSECURE!)
            # But we'll just tell them to contact support
            return {
                'success': True,
                'message': f"Please contact sales@octainsight.com for password reset assistance.",
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
    
    def check_user_limits(self, user_id: str) -> Dict[str, Any]:
        """Check if user can create more projects/upload files"""
        try:
            user = self.client.table('users').select('*').eq('user_id', user_id).execute()
            
            if not user.data:
                return {
                    'can_create_project': False,
                    'can_upload': False,
                    'reason': 'User not found'
                }
            
            user_data = user.data[0]
            
            can_create = user_data.get('current_project_count', 0) < user_data.get('max_projects', 3)
            can_upload = user_data.get('uploads_this_month', 0) < user_data.get('max_uploads_per_month', 50)
            
            return {
                'can_create_project': can_create,
                'can_upload': can_upload,
                'current_projects': user_data.get('current_project_count', 0),
                'max_projects': user_data.get('max_projects', 3),
                'current_uploads': user_data.get('uploads_this_month', 0),
                'max_uploads': user_data.get('max_uploads_per_month', 50)
            }
            
        except Exception as e:
            st.error(f"Error checking limits: {str(e)}")
            return {
                'can_create_project': False,
                'can_upload': False,
                'reason': str(e)
            }
    
    def create_project(self, owner_id: str, project_name: str, 
                      description: str = None, baseline_year: int = None,
                      scenario_target_year: int = None) -> Optional[Dict]:
        """Create new project"""
        try:
            # Generate project code
            import random
            import string
            project_code = f"PRJ-{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"
            
            project_data = {
                'owner_id': owner_id,
                'project_name': project_name,
                'project_code': project_code,
                'description': description,
                'status': 'active',
                'workflow_state': 'setup',
                'current_page': 2,
                'completion_percentage': 0
            }
            
            # Only add year fields if provided
            if baseline_year is not None:
                project_data['baseline_year'] = baseline_year
            if scenario_target_year is not None:
                project_data['scenario_target_year'] = scenario_target_year
            
            result = self.client.table('projects').insert(project_data).execute()
            
            if result.data:
                # Update user's project count
                user = self.client.table('users').select('current_project_count').eq('user_id', owner_id).execute()
                if user.data:
                    new_count = user.data[0].get('current_project_count', 0) + 1
                    self.client.table('users').update({
                        'current_project_count': new_count
                    }).eq('user_id', owner_id).execute()
                
                return result.data[0]
            
            return None
            
        except Exception as e:
            st.error(f"Error creating project: {str(e)}")
            return None
    
    def delete_project(self, project_id: str, user_id: str) -> bool:
        """Delete project (soft delete by setting status to archived)"""
        try:
            # Verify ownership
            project = self.client.table('projects').select('owner_id').eq('project_id', project_id).execute()
            
            if not project.data or project.data[0]['owner_id'] != user_id:
                st.error("You don't have permission to delete this project")
                return False
            
            # Soft delete - set status to archived
            self.client.table('projects').update({
                'status': 'archived',
                'updated_at': datetime.now().isoformat()
            }).eq('project_id', project_id).execute()
            
            # Update user's project count
            user = self.client.table('users').select('current_project_count').eq('user_id', user_id).execute()
            if user.data:
                new_count = max(0, user.data[0].get('current_project_count', 0) - 1)
                self.client.table('users').update({
                    'current_project_count': new_count
                }).eq('user_id', user_id).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error deleting project: {str(e)}")
            return False


# Singleton pattern with caching
@st.cache_resource
def get_db_manager() -> SupabaseManager:
    """Get cached database manager instance"""
    return SupabaseManager()
