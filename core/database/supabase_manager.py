"""
ExplainFutures - Supabase Database Manager
Complete CRUD operations for all tables
"""

import streamlit as st
from supabase import create_client, Client
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime, timedelta
import json
import bcrypt


class SupabaseManager:
    """
    Central database manager for ExplainFutures with Supabase
    
    Usage:
        db = SupabaseManager()
        user = db.create_user("john_doe", "john@example.com", "password123")
        projects = db.get_user_projects(user['user_id'])
    """
    
    def __init__(self):
        """Initialize Supabase client from Streamlit secrets"""
        try:
            # Get credentials from Streamlit secrets (only 2 parameters needed)
            self.url = st.secrets["supabase"]["url"]
            self.key = st.secrets["supabase"]["key"]
            
            # Create Supabase client (same client for all operations)
            self.client: Client = create_client(self.url, self.key)
            
            # Demo user/project IDs
            self.demo_user_id = st.secrets["app"]["demo_user_id"]
            self.demo_project_id = st.secrets["app"]["demo_project_id"]
            
        except Exception as e:
            st.error(f"❌ Failed to connect to Supabase: {str(e)}")
            st.info("Please check your Streamlit secrets configuration")
            raise
    
    # ========================================================================
    # USER MANAGEMENT
    # ========================================================================
    
    def create_user(self, username: str, email: str, password: str, 
                   full_name: str = None, **kwargs) -> Dict:
        """
        Create a new user
        
        Args:
            username: Unique username
            email: User email
            password: Plain text password (will be hashed)
            full_name: User's full name
            **kwargs: Additional user fields
        
        Returns:
            Created user dict
        """
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        user_data = {
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'full_name': full_name,
            **kwargs
        }
        
        response = self.client.table('users').insert(user_data).execute()
        
        if response.data:
            self.log_audit(None, response.data[0]['user_id'], 'create', 'user', response.data[0]['user_id'])
            return response.data[0]
        
        return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        response = self.client.table('users').select('*').eq('username', username).execute()
        return response.data[0] if response.data else None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        response = self.client.table('users').select('*').eq('email', email).execute()
        return response.data[0] if response.data else None
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verify user password"""
        user = self.get_user_by_username(username)
        if not user:
            return False
        
        return bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8'))
    
    def login_user(self, username: str, password: str, ip_address: str = None, 
                  user_agent: str = None) -> Optional[Dict]:
        """
        Login user and create session
        
        Returns:
            User dict if successful, None if failed
        """
        user = self.get_user_by_username(username)
        
        if not user or not user['is_active']:
            self.log_login_attempt(user['user_id'] if user else None, False, "User inactive or not found", ip_address, user_agent)
            return None
        
        if not self.verify_password(username, password):
            self.log_login_attempt(user['user_id'], False, "Invalid password", ip_address, user_agent)
            
            # Increment failed attempts
            self.client.table('users').update({
                'failed_login_attempts': user.get('failed_login_attempts', 0) + 1,
                'last_failed_login': datetime.now().isoformat()
            }).eq('user_id', user['user_id']).execute()
            
            return None
        
        # Update last login
        self.client.table('users').update({
            'last_login': datetime.now().isoformat(),
            'last_login_ip': ip_address,
            'last_login_user_agent': user_agent,
            'login_count': user.get('login_count', 0) + 1,
            'failed_login_attempts': 0
        }).eq('user_id', user['user_id']).execute()
        
        # Log successful login
        self.log_login_attempt(user['user_id'], True, None, ip_address, user_agent)
        
        return user
    
    def log_login_attempt(self, user_id: str, successful: bool, failure_reason: str = None,
                         ip_address: str = None, user_agent: str = None):
        """Log login attempt"""
        self.client.table('user_login_history').insert({
            'user_id': user_id,
            'login_successful': successful,
            'failure_reason': failure_reason,
            'ip_address': ip_address,
            'user_agent': user_agent
        }).execute()
    
    def is_demo_user(self, user_id: str) -> bool:
        """Check if user is demo user"""
        return str(user_id) == str(self.demo_user_id)
    
    # ========================================================================
    # PROJECT MANAGEMENT
    # ========================================================================
    
    def create_project(self, owner_id: str, project_name: str, description: str = None,
                      application_name: str = 'explainfutures', **kwargs) -> Dict:
        """Create a new project"""
        project_data = {
            'project_name': project_name,
            'description': description,
            'owner_id': owner_id,
            'application_name': application_name,
            'project_code': self._generate_project_code(),
            **kwargs
        }
        
        response = self.client.table('projects').insert(project_data).execute()
        
        if response.data:
            project_id = response.data[0]['project_id']
            
            # Update user project count
            user = self.client.table('users').select('current_project_count').eq('user_id', owner_id).execute()
            if user.data:
                new_count = user.data[0].get('current_project_count', 0) + 1
                self.client.table('users').update({'current_project_count': new_count}).eq('user_id', owner_id).execute()
            
            self.log_audit(project_id, owner_id, 'create', 'project', project_id)
            return response.data[0]
        
        return None
    
    def get_user_projects(self, user_id: str, include_collaborations: bool = True) -> List[Dict]:
        """Get all projects user has access to"""
        # Get owned projects
        owned = self.client.table('projects').select('*').eq('owner_id', user_id).eq('deleted_at', None).execute()
        projects = owned.data if owned.data else []
        
        if include_collaborations:
            # Get collaborated projects
            collab_response = self.client.table('project_collaborators').select(
                'project_id, role, can_view, can_edit'
            ).eq('user_id', user_id).eq('invitation_status', 'accepted').execute()
            
            if collab_response.data:
                collab_project_ids = [c['project_id'] for c in collab_response.data]
                
                for project_id in collab_project_ids:
                    proj = self.client.table('projects').select('*').eq('project_id', project_id).execute()
                    if proj.data:
                        projects.append(proj.data[0])
        
        return projects
    
    def get_project(self, project_id: str) -> Optional[Dict]:
        """Get project by ID"""
        response = self.client.table('projects').select('*').eq('project_id', project_id).execute()
        return response.data[0] if response.data else None
    
    def update_project_progress(self, project_id: str, workflow_state: str = None,
                               current_page: int = None, completion_percentage: int = None):
        """Update project workflow progress"""
        update_data = {}
        
        if workflow_state:
            update_data['workflow_state'] = workflow_state
        if current_page is not None:
            update_data['current_page'] = current_page
        if completion_percentage is not None:
            update_data['completion_percentage'] = completion_percentage
        
        update_data['last_accessed'] = datetime.now().isoformat()
        
        self.client.table('projects').update(update_data).eq('project_id', project_id).execute()
    
    def delete_project(self, project_id: str, user_id: str, hard_delete: bool = False):
        """Delete project (soft delete by default)"""
        if hard_delete:
            self.client.table('projects').delete().eq('project_id', project_id).execute()
        else:
            self.client.table('projects').update({
                'deleted_at': datetime.now().isoformat(),
                'status': 'deleted'
            }).eq('project_id', project_id).execute()
        
        self.log_audit(project_id, user_id, 'delete', 'project', project_id)
    
    def _generate_project_code(self) -> str:
        """Generate unique project code"""
        timestamp = datetime.now().strftime('%Y%m%d')
        random_suffix = str(uuid.uuid4())[:6].upper()
        return f"PROJ-{timestamp}-{random_suffix}"
    
    # ========================================================================
    # DEMO SESSION MANAGEMENT
    # ========================================================================
    
    def create_demo_session(self, user_id: str, project_id: str, 
                           duration_seconds: int = 1800) -> Dict:
        """
        Create a new demo session
        
        Args:
            user_id: User ID (can be demo user or regular user)
            project_id: Demo project ID
            duration_seconds: Session duration (default 30 min)
        
        Returns:
            Created session dict
        """
        session_token = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(seconds=duration_seconds)
        
        session_data = {
            'user_id': user_id,
            'project_id': project_id,
            'session_token': session_token,
            'expires_at': expires_at.isoformat(),
            'ip_address': st.context.headers.get('X-Forwarded-For') if hasattr(st, 'context') else None,
            'user_agent': st.context.headers.get('User-Agent') if hasattr(st, 'context') else None
        }
        
        response = self.client.table('demo_sessions').insert(session_data).execute()
        return response.data[0] if response.data else None
    
    def get_active_demo_session(self, user_id: str) -> Optional[Dict]:
        """Get user's active demo session"""
        response = self.client.table('demo_sessions').select('*').eq(
            'user_id', user_id
        ).eq('cleanup_completed', False).gte(
            'expires_at', datetime.now().isoformat()
        ).order('started_at', desc=True).limit(1).execute()
        
        return response.data[0] if response.data else None
    
    def end_demo_session(self, session_id: str):
        """End demo session and trigger cleanup"""
        # Call cleanup function
        self.client.rpc('cleanup_demo_session', {'p_session_id': session_id}).execute()
    
    def track_demo_action(self, session_id: str, entity_type: str, entity_id: str):
        """Track entity created in demo session"""
        session = self.client.table('demo_sessions').select('*').eq('session_id', session_id).execute()
        
        if session.data:
            field_name = f'created_{entity_type}s'  # e.g., 'created_files'
            current_list = session.data[0].get(field_name, [])
            
            if not isinstance(current_list, list):
                current_list = []
            
            current_list.append(entity_id)
            
            self.client.table('demo_sessions').update({
                field_name: current_list
            }).eq('session_id', session_id).execute()
    
    # ========================================================================
    # PARAMETER MANAGEMENT
    # ========================================================================
    
    def save_parameter(self, project_id: str, created_by: str, name: str,
                      timestamps: List, values: List, source: str,
                      demo_session_id: str = None, **kwargs) -> Dict:
        """Save or update parameter"""
        param_data = {
            'project_id': project_id,
            'created_by': created_by,
            'name': name,
            'timestamps': json.dumps(timestamps) if isinstance(timestamps, list) else timestamps,
            'values': json.dumps(values) if isinstance(values, list) else values,
            'source': source,
            'demo_session_id': demo_session_id,
            'is_demo_data': demo_session_id is not None,
            **kwargs
        }
        
        response = self.client.table('parameters').insert(param_data).execute()
        
        if response.data and demo_session_id:
            self.track_demo_action(demo_session_id, 'parameter', response.data[0]['parameter_id'])
        
        return response.data[0] if response.data else None
    
    def get_parameters(self, project_id: str, source: str = None, 
                      category: str = None, include_demo_base: bool = True) -> List[Dict]:
        """Get parameters for project"""
        query = self.client.table('parameters').select('*').eq('project_id', project_id)
        
        if source:
            query = query.eq('source', source)
        if category:
            query = query.eq('category', category)
        if not include_demo_base:
            query = query.eq('is_base_data', False)
        
        response = query.execute()
        
        # Parse JSON fields
        if response.data:
            for param in response.data:
                if isinstance(param.get('timestamps'), str):
                    param['timestamps'] = json.loads(param['timestamps'])
                if isinstance(param.get('values'), str):
                    param['values'] = json.loads(param['values'])
        
        return response.data if response.data else []
    
    # ========================================================================
    # MODEL MANAGEMENT
    # ========================================================================
    
    def save_trained_model(self, project_id: str, parameter_id: str, created_by: str,
                          model_name: str, test_metrics: Dict, model_blob: bytes = None,
                          demo_session_id: str = None, **kwargs) -> Dict:
        """Save trained model"""
        model_data = {
            'project_id': project_id,
            'parameter_id': parameter_id,
            'created_by': created_by,
            'model_name': model_name,
            'test_metrics': json.dumps(test_metrics),
            'model_blob': model_blob,
            'demo_session_id': demo_session_id,
            'is_demo_data': demo_session_id is not None,
            **kwargs
        }
        
        response = self.client.table('trained_models').insert(model_data).execute()
        
        if response.data and demo_session_id:
            self.track_demo_action(demo_session_id, 'model', response.data[0]['model_id'])
        
        return response.data[0] if response.data else None
    
    def get_models(self, project_id: str, parameter_id: str = None) -> List[Dict]:
        """Get trained models"""
        query = self.client.table('trained_models').select('*').eq('project_id', project_id)
        
        if parameter_id:
            query = query.eq('parameter_id', parameter_id)
        
        response = query.execute()
        return response.data if response.data else []
    
    # ========================================================================
    # SCENARIO MANAGEMENT
    # ========================================================================
    
    def save_scenario(self, project_id: str, created_by: str, title: str,
                     description: str = None, demo_session_id: str = None,
                     **kwargs) -> Dict:
        """Save scenario"""
        scenario_data = {
            'project_id': project_id,
            'created_by': created_by,
            'title': title,
            'description': description,
            'demo_session_id': demo_session_id,
            'is_demo_data': demo_session_id is not None,
            **kwargs
        }
        
        response = self.client.table('scenarios').insert(scenario_data).execute()
        
        if response.data and demo_session_id:
            self.track_demo_action(demo_session_id, 'scenario', response.data[0]['scenario_id'])
        
        return response.data[0] if response.data else None
    
    def save_scenario_parameter(self, scenario_id: str, project_id: str,
                               parameter_name: str, value: float, unit: str,
                               direction: str, demo_session_id: str = None,
                               **kwargs) -> Dict:
        """Save scenario parameter"""
        param_data = {
            'scenario_id': scenario_id,
            'project_id': project_id,
            'parameter_name': parameter_name,
            'value': value,
            'unit': unit,
            'direction': direction,
            'demo_session_id': demo_session_id,
            'is_demo_data': demo_session_id is not None,
            **kwargs
        }
        
        response = self.client.table('scenario_parameters').insert(param_data).execute()
        return response.data[0] if response.data else None
    
    def get_scenarios(self, project_id: str, include_demo_base: bool = True) -> List[Dict]:
        """Get scenarios for project"""
        query = self.client.table('scenarios').select('*').eq('project_id', project_id)
        
        if not include_demo_base:
            query = query.eq('is_base_data', False)
        
        response = query.execute()
        return response.data if response.data else []
    
    # ========================================================================
    # AUDIT & LOGGING
    # ========================================================================
    
    def log_audit(self, project_id: str, user_id: str, event_type: str,
                 entity_type: str, entity_id: str, action: str = None,
                 old_value: Dict = None, new_value: Dict = None):
        """Log audit event"""
        audit_data = {
            'project_id': project_id,
            'user_id': user_id,
            'event_type': event_type,
            'entity_type': entity_type,
            'entity_id': entity_id,
            'action': action,
            'old_value': json.dumps(old_value) if old_value else None,
            'new_value': json.dumps(new_value) if new_value else None
        }
        
        self.client.table('audit_log').insert(audit_data).execute()
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def check_user_limits(self, user_id: str) -> Dict[str, bool]:
        """Check if user has reached limits"""
        user = self.client.table('users').select('*').eq('user_id', user_id).execute()
        
        if not user.data:
            return {'can_create_project': False, 'can_upload': False}
        
        user_data = user.data[0]
        
        return {
            'can_create_project': user_data.get('current_project_count', 0) < user_data.get('max_projects', 3),
            'can_upload': user_data.get('uploads_this_month', 0) < user_data.get('max_uploads_per_month', 50),
            'can_use_storage': user_data.get('current_storage_mb', 0) < user_data.get('max_storage_mb', 1000)
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

@st.cache_resource
def get_db_manager() -> SupabaseManager:
    """
    Get cached database manager instance
    
    Usage:
        from core.database.supabase_manager import get_db_manager
        
        db = get_db_manager()
        user = db.login_user("demo", "demo123")
    """
    return SupabaseManager()
