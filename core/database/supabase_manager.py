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
import pandas as pd


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
    
    def convert_timestamps_to_serializable(self, obj):
        """
        Recursively convert pandas Timestamps and datetime objects to ISO format strings
        for JSON serialization
        """
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self.convert_timestamps_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_timestamps_to_serializable(item) for item in obj]
        else:
            return obj
    
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
    
    def get_user_projects(self, user_id: str, include_collaborations: bool = True, include_deleted: bool = False) -> List[Dict]:
        """Get user's projects (owned + collaborated)"""
        try:
            # Build query for owned projects
            owned_query = self.client.table('projects').select('*').eq('owner_id', user_id)
            
            # Exclude deleted projects unless specifically requested
            if not include_deleted:
                owned_query = owned_query.neq('status', 'deleted')
            
            owned_result = owned_query.execute()
            owned_projects = owned_result.data if owned_result.data else []
            
            # Mark as owned
            for project in owned_projects:
                project['access_role'] = 'owner'
                project['is_owner'] = True
            
            if not include_collaborations:
                return owned_projects
            
            try:
                # Get collaborated projects
                collab_query = self.client.table('project_collaborators').select(
                    'project_id, role, can_edit, can_delete, created_at'
                ).eq('user_id', user_id)
                collab_result = collab_query.execute()
                
                if collab_result.data:
                    # Get project details for collaborated projects
                    project_ids = [c['project_id'] for c in collab_result.data]
                    
                    if project_ids:
                        projects_query = self.client.table('projects').select('*').in_('project_id', project_ids)
                        
                        # Exclude deleted projects from collaborations
                        if not include_deleted:
                            projects_query = projects_query.neq('status', 'deleted')
                        
                        projects_result = projects_query.execute()
                        
                        if projects_result.data:
                            # Add collaboration info to projects
                            collab_dict = {c['project_id']: c for c in collab_result.data}
                            
                            for project in projects_result.data:
                                collab_info = collab_dict.get(project['project_id'])
                                if collab_info:
                                    project['access_role'] = collab_info.get('role', 'collaborator')
                                    project['is_owner'] = False
                                    project['can_edit'] = collab_info.get('can_edit', True)
                                    project['can_delete'] = collab_info.get('can_delete', False)
                                    project['collaboration_since'] = collab_info.get('created_at')
                            
                            owned_projects.extend(projects_result.data)
            
            except Exception as collab_error:
                # If collaboration fetch fails, just return owned projects
                st.warning(f"Note: Could not fetch shared projects: {str(collab_error)}")
            
            return owned_projects
            
        except Exception as e:
            st.error(f"Error fetching projects: {str(e)}")
            return []
    
    def get_project_collaborators(self, project_id: str) -> List[Dict]:
        """Get all collaborators for a project with their user info"""
        try:
            # Specify the exact foreign key relationship to use
            # Using the user_id foreign key (not invited_by)
            result = self.client.table('project_collaborators').select(
                '*, users!project_collaborators_user_id_fkey(user_id, username, email, full_name)'
            ).eq('project_id', project_id).execute()
            
            if result.data:
                # Flatten the nested user data
                collaborators = []
                for collab in result.data:
                    user_info = collab.get('users', {})
                    if user_info:  # Only add if user info exists
                        collaborators.append({
                            'collaborator_id': collab.get('collaborator_id'),
                            'user_id': collab['user_id'],
                            'username': user_info.get('username', 'Unknown'),
                            'email': user_info.get('email', ''),
                            'full_name': user_info.get('full_name', user_info.get('username', 'Unknown')),
                            'role': collab.get('role', 'collaborator'),
                            'can_edit': collab.get('can_edit', True),
                            'can_delete': collab.get('can_delete', False),
                            'created_at': collab.get('created_at')
                        })
                return collaborators
            
            return []
            
        except Exception as e:
            # If the specific foreign key doesn't work, try a simpler approach
            try:
                # Get collaborators without join
                collab_result = self.client.table('project_collaborators').select(
                    'collaborator_id, user_id, role, can_edit, can_delete, created_at'
                ).eq('project_id', project_id).execute()
                
                if not collab_result.data:
                    return []
                
                # Get user info separately
                collaborators = []
                for collab in collab_result.data:
                    user_result = self.client.table('users').select(
                        'user_id, username, email, full_name'
                    ).eq('user_id', collab['user_id']).execute()
                    
                    if user_result.data:
                        user_info = user_result.data[0]
                        collaborators.append({
                            'collaborator_id': collab.get('collaborator_id'),
                            'user_id': collab['user_id'],
                            'username': user_info.get('username', 'Unknown'),
                            'email': user_info.get('email', ''),
                            'full_name': user_info.get('full_name', user_info.get('username', 'Unknown')),
                            'role': collab.get('role', 'collaborator'),
                            'can_edit': collab.get('can_edit', True),
                            'can_delete': collab.get('can_delete', False),
                            'created_at': collab.get('created_at')
                        })
                
                return collaborators
                
            except Exception as e2:
                st.error(f"Error fetching collaborators: {str(e2)}")
                return []
    
    def remove_collaborator(self, project_id: str, user_id: str, requesting_user_id: str) -> bool:
        """Remove a collaborator from a project (owner only)"""
        try:
            # Verify requesting user is the owner
            project = self.client.table('projects').select('owner_id').eq('project_id', project_id).execute()
            
            if not project.data or project.data[0]['owner_id'] != requesting_user_id:
                st.error("Only the project owner can remove collaborators")
                return False
            
            # Remove collaborator
            self.client.table('project_collaborators').delete().eq(
                'project_id', project_id
            ).eq('user_id', user_id).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error removing collaborator: {str(e)}")
            return False
    
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
        """
        Soft delete project (marks as deleted, doesn't remove from database)
        Also removes all collaborators when project is deleted
        """
        try:
            # Verify ownership
            project = self.client.table('projects').select('owner_id, project_name').eq('project_id', project_id).execute()
            
            if not project.data or project.data[0]['owner_id'] != user_id:
                st.error("Only the project owner can delete this project")
                return False
            
            project_name = project.data[0].get('project_name', 'Unknown')
            
            # Soft delete - set status to deleted (not archived)
            self.client.table('projects').update({
                'status': 'deleted',
                'deleted_at': datetime.now().isoformat(),
                'deleted_by': user_id,
                'updated_at': datetime.now().isoformat()
            }).eq('project_id', project_id).execute()
            
            # Remove all collaborators when project is deleted
            try:
                self.client.table('project_collaborators').delete().eq('project_id', project_id).execute()
            except Exception as e:
                st.warning(f"Note: Could not remove collaborators: {str(e)}")
            
            # Update user's project count (only count active projects)
            try:
                # Count active projects
                active_projects = self.client.table('projects').select(
                    'project_id', count='exact'
                ).eq('owner_id', user_id).eq('status', 'active').execute()
                
                new_count = active_projects.count if hasattr(active_projects, 'count') else 0
                
                self.client.table('users').update({
                    'current_project_count': new_count
                }).eq('user_id', user_id).execute()
            except Exception as e:
                st.warning(f"Note: Could not update project count: {str(e)}")
            
            st.success(f"✅ Project '{project_name}' moved to trash")
            st.info("💡 Project can be restored from the trash within 30 days")
            return True
            
        except Exception as e:
            st.error(f"Error deleting project: {str(e)}")
            return False
    
    def restore_project(self, project_id: str, user_id: str) -> bool:
        """Restore a soft-deleted project"""
        try:
            # Verify ownership
            project = self.client.table('projects').select('owner_id, status, project_name').eq('project_id', project_id).execute()
            
            if not project.data:
                st.error("Project not found")
                return False
            
            if project.data[0]['owner_id'] != user_id:
                st.error("Only the project owner can restore this project")
                return False
            
            if project.data[0]['status'] != 'deleted':
                st.warning("Project is not deleted")
                return False
            
            project_name = project.data[0].get('project_name', 'Unknown')
            
            # Restore project
            self.client.table('projects').update({
                'status': 'active',
                'deleted_at': None,
                'deleted_by': None,
                'updated_at': datetime.now().isoformat()
            }).eq('project_id', project_id).execute()
            
            # Update user's project count
            try:
                active_projects = self.client.table('projects').select(
                    'project_id', count='exact'
                ).eq('owner_id', user_id).eq('status', 'active').execute()
                
                new_count = active_projects.count if hasattr(active_projects, 'count') else 0
                
                self.client.table('users').update({
                    'current_project_count': new_count
                }).eq('user_id', user_id).execute()
            except:
                pass
            
            st.success(f"✅ Project '{project_name}' restored successfully")
            return True
            
        except Exception as e:
            st.error(f"Error restoring project: {str(e)}")
            return False
    
    def permanently_delete_project(self, project_id: str, user_id: str) -> bool:
        """
        PERMANENTLY delete a project from database (admin only)
        WARNING: This cannot be undone!
        """
        try:
            # Verify ownership
            project = self.client.table('projects').select('owner_id').eq('project_id', project_id).execute()
            
            if not project.data or project.data[0]['owner_id'] != user_id:
                st.error("Unauthorized")
                return False
            
            # Delete all related data first
            self.client.table('project_collaborators').delete().eq('project_id', project_id).execute()
            
            # Finally delete the project
            self.client.table('projects').delete().eq('project_id', project_id).execute()
            
            # Update user's project count
            user = self.client.table('users').select('current_project_count').eq('user_id', user_id).execute()
            if user.data:
                new_count = max(0, user.data[0].get('current_project_count', 0) - 1)
                self.client.table('users').update({
                    'current_project_count': new_count
                }).eq('user_id', user_id).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error permanently deleting project: {str(e)}")
            return False
    
    def rename_project(self, project_id: str, new_name: str, user_id: str) -> bool:
        """Rename a project (owner only)"""
        try:
            # Verify ownership
            project = self.client.table('projects').select('owner_id, project_name').eq('project_id', project_id).execute()
            
            if not project.data:
                st.error("Project not found")
                return False
            
            if project.data[0]['owner_id'] != user_id:
                st.error("Only the project owner can rename this project")
                return False
            
            old_name = project.data[0].get('project_name', 'Unknown')
            
            # Validate new name
            if not new_name or len(new_name.strip()) == 0:
                st.error("Project name cannot be empty")
                return False
            
            if len(new_name) > 200:
                st.error("Project name is too long (max 200 characters)")
                return False
            
            # Update project name
            self.client.table('projects').update({
                'project_name': new_name.strip(),
                'updated_at': datetime.now().isoformat()
            }).eq('project_id', project_id).execute()
            
            st.success(f"✅ Project renamed from '{old_name}' to '{new_name}'")
            return True
            
        except Exception as e:
            st.error(f"Error renaming project: {str(e)}")
            return False
    
    # ========================================================================
    # FILE UPLOAD MANAGEMENT
    # ========================================================================
    
    def save_uploaded_file(self, project_id: str, filename: str, file_size: int, 
                          file_type: str, metadata: dict = None) -> Optional[Dict]:
        """Save uploaded file information to database"""
        try:
            result = self.client.table('uploaded_files').insert({
                'project_id': project_id,
                'filename': filename,
                'file_size': file_size,
                'file_type': file_type,
                'metadata': json.dumps(metadata) if metadata else None,
                'uploaded_at': datetime.now().isoformat()
            }).execute()
            
            return result.data[0] if result.data else None
            
        except Exception as e:
            st.error(f"Error saving file info: {str(e)}")
            return None
    
    def get_uploaded_files(self, project_id: str) -> List[Dict]:
        """Get all uploaded files for a project"""
        try:
            result = self.client.table('uploaded_files').select('*').eq(
                'project_id', project_id
            ).order('uploaded_at', desc=True).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            st.error(f"Error fetching uploaded files: {str(e)}")
            return []
    
    # ========================================================================
    # PARAMETER MANAGEMENT
    # ========================================================================
    
    def save_parameters(self, project_id: str, parameters: List[Dict]) -> bool:
        """Save or update parameters for a project"""
        try:
            for param in parameters:
                # Check if parameter already exists
                existing = self.client.table('parameters').select('parameter_id').eq(
                    'project_id', project_id
                ).eq('parameter_name', param['name']).execute()
                
                param_data = {
                    'project_id': project_id,
                    'parameter_name': param['name'],
                    'data_type': param.get('data_type', 'numeric'),
                    'unit': param.get('unit'),
                    'description': param.get('description'),
                    'min_value': param.get('min_value'),
                    'max_value': param.get('max_value'),
                    'mean_value': param.get('mean_value'),
                    'std_value': param.get('std_value'),
                    'missing_count': param.get('missing_count', 0),
                    'total_count': param.get('total_count', 0),
                    'updated_at': datetime.now().isoformat()
                }
                
                if existing.data:
                    # Update existing parameter
                    self.client.table('parameters').update(param_data).eq(
                        'parameter_id', existing.data[0]['parameter_id']
                    ).execute()
                else:
                    # Insert new parameter
                    param_data['created_at'] = datetime.now().isoformat()
                    self.client.table('parameters').insert(param_data).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error saving parameters: {str(e)}")
            return False
    
    def get_project_parameters(self, project_id: str) -> List[Dict]:
        """Get all parameters for a project"""
        try:
            result = self.client.table('parameters').select('*').eq(
                'project_id', project_id
            ).order('parameter_name').execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            st.error(f"Error fetching parameters: {str(e)}")
            return []
    
    def check_duplicate_parameters(self, project_id: str) -> Dict[str, List[Dict]]:
        """Check for duplicate parameter names in a project"""
        try:
            # Get all parameters
            params = self.get_project_parameters(project_id)
            
            # Group by parameter name
            duplicates = {}
            param_groups = {}
            
            for param in params:
                name = param['parameter_name']
                if name not in param_groups:
                    param_groups[name] = []
                param_groups[name].append(param)
            
            # Find duplicates (count > 1)
            for name, group in param_groups.items():
                if len(group) > 1:
                    duplicates[name] = group
            
            return duplicates
            
        except Exception as e:
            st.error(f"Error checking duplicates: {str(e)}")
            return {}
    
    def merge_parameters(self, parameter_ids: List[str], keep_id: str) -> bool:
        """Merge duplicate parameters (keep one, delete others)"""
        try:
            # Delete all except the one to keep
            for param_id in parameter_ids:
                if param_id != keep_id:
                    self.client.table('parameters').delete().eq('parameter_id', param_id).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error merging parameters: {str(e)}")
            return False
    
    def rename_parameter(self, parameter_id: str, new_name: str) -> bool:
        """Rename a parameter"""
        try:
            self.client.table('parameters').update({
                'parameter_name': new_name,
                'updated_at': datetime.now().isoformat()
            }).eq('parameter_id', parameter_id).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error renaming parameter: {str(e)}")
            return False
    
    def delete_parameter(self, parameter_id: str) -> bool:
        """Delete a parameter"""
        try:
            self.client.table('parameters').delete().eq('parameter_id', parameter_id).execute()
            return True
            
        except Exception as e:
            st.error(f"Error deleting parameter: {str(e)}")
            return False
    
    # ========================================================================
    # PROJECT STEP TRACKING
    # ========================================================================
    
    def update_step_completion(self, project_id: str, step_key: str, completed: bool = True) -> bool:
        """Update completion status of a specific workflow step"""
        try:
            # Get current step completion data
            result = self.client.table('projects').select('step_completion').eq('project_id', project_id).execute()
            
            if not result.data:
                return False
            
            # Get existing completion data or initialize empty dict
            step_completion = result.data[0].get('step_completion', {})
            if step_completion is None:
                step_completion = {}
            
            # Update the specific step
            step_completion[step_key] = completed
            
            # Save back to database
            self.client.table('projects').update({
                'step_completion': step_completion,
                'updated_at': datetime.now().isoformat()
            }).eq('project_id', project_id).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error updating step completion: {str(e)}")
            return False
    
    def get_step_completion(self, project_id: str) -> Dict[str, bool]:
        """Get completion status of all workflow steps"""
        try:
            result = self.client.table('projects').select('step_completion').eq('project_id', project_id).execute()
            
            if result.data and result.data[0].get('step_completion'):
                return result.data[0]['step_completion']
            
            return {}
            
        except Exception as e:
            st.error(f"Error fetching step completion: {str(e)}")
            return {}
    
    def load_project_data_for_health_report(self, project_id: str) -> Dict[str, Any]:
        """Load all project data needed for health report from database"""
        try:
            # Get parameters
            parameters = self.get_project_parameters(project_id)
            
            if not parameters:
                return {
                    'success': False,
                    'message': 'No parameters found. Please upload data first.',
                    'parameters': [],
                    'has_data': False
                }
            
            # Convert parameters to DataFrame-like structure for health report
            # This simulates what would be in session state
            param_summary = {
                'parameters': parameters,
                'variable_count': len(parameters),
                'has_data': True,
                'success': True
            }
            
            # Calculate aggregate statistics
            total_missing = sum(p.get('missing_count', 0) for p in parameters)
            total_count = sum(p.get('total_count', 0) for p in parameters)
            
            param_summary['total_missing'] = total_missing
            param_summary['total_count'] = total_count
            param_summary['overall_missing_pct'] = total_missing / total_count if total_count > 0 else 0
            
            return param_summary
            
        except Exception as e:
            st.error(f"Error loading project data: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'parameters': [],
                'has_data': False
            }
    
    # ========================================================================
    # HEALTH REPORT MANAGEMENT
    # ========================================================================
    
    def save_health_report(self, project_id: str, health_data: Dict[str, Any]) -> bool:
        """Save or update health report for a project"""
        try:
            # Calculate data hash from parameters
            import hashlib
            parameters = self.get_project_parameters(project_id)
            param_string = json.dumps(sorted([p['parameter_name'] for p in parameters]))
            data_hash = hashlib.md5(param_string.encode()).hexdigest()
            
            # Convert ALL potential Timestamp/datetime objects before json.dumps
            missing_values_detail = self.convert_timestamps_to_serializable(
                health_data.get('missing_values_detail', {})
            )
            outliers_detail = self.convert_timestamps_to_serializable(
                health_data.get('outliers_detail', {})
            )
            coverage_detail = self.convert_timestamps_to_serializable(
                health_data.get('coverage_detail', {})
            )
            issues_list = self.convert_timestamps_to_serializable(
                health_data.get('issues_list', [])
            )
            time_metadata = self.convert_timestamps_to_serializable(
                health_data.get('time_metadata', {})
            )
            parameters_analyzed = self.convert_timestamps_to_serializable(
                health_data.get('parameters_analyzed', [])
            )
            
            # Helper function to ensure native Python types (not numpy)
            def to_python_int(value):
                """Convert to native Python int, handling numpy types"""
                if value is None:
                    return 0
                # Handle numpy types
                if hasattr(value, 'item'):
                    value = value.item()
                return int(float(value))
            
            def to_python_float(value):
                """Convert to native Python float, handling numpy types"""
                if value is None:
                    return 0.0
                # Handle numpy types
                if hasattr(value, 'item'):
                    value = value.item()
                return float(value)
            
            report_data = {
                'project_id': str(project_id),
                'health_score': to_python_int(health_data.get('health_score', 0)),
                'health_category': str(health_data.get('health_category', 'poor')),
                'total_parameters': to_python_int(health_data.get('total_parameters', 0)),
                'total_data_points': to_python_int(health_data.get('total_data_points', 0)),
                'total_missing_values': to_python_int(health_data.get('total_missing_values', 0)),
                'missing_percentage': to_python_float(health_data.get('missing_percentage', 0)),
                'critical_issues': to_python_int(health_data.get('critical_issues', 0)),
                'warnings': to_python_int(health_data.get('warnings', 0)),
                'duplicate_timestamps': to_python_int(health_data.get('duplicate_timestamps', 0)),
                'outlier_count': to_python_int(health_data.get('outlier_count', 0)),
                'missing_values_detail': json.dumps(missing_values_detail),
                'outliers_detail': json.dumps(outliers_detail),
                'coverage_detail': json.dumps(coverage_detail),
                'issues_list': json.dumps(issues_list),
                'time_metadata': json.dumps(time_metadata),
                'parameters_analyzed': parameters_analyzed,
                'data_hash': str(data_hash),
                'updated_at': datetime.now().isoformat()
            }
            
            # Check if report exists
            existing = self.client.table('health_reports').select('report_id').eq(
                'project_id', project_id
            ).order('created_at', desc=True).limit(1).execute()
            
            if existing.data:
                # Update existing report
                self.client.table('health_reports').update(report_data).eq(
                    'report_id', existing.data[0]['report_id']
                ).execute()
            else:
                # Create new report
                self.client.table('health_reports').insert(report_data).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error saving health report: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_health_report(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest health report for a project"""
        try:
            result = self.client.table('health_reports').select('*').eq(
                'project_id', project_id
            ).order('created_at', desc=True).limit(1).execute()
            
            if result.data:
                report = result.data[0]
                
                # Parse JSON fields
                report['missing_values_detail'] = json.loads(report.get('missing_values_detail', '{}'))
                report['outliers_detail'] = json.loads(report.get('outliers_detail', '{}'))
                report['coverage_detail'] = json.loads(report.get('coverage_detail', '{}'))
                report['issues_list'] = json.loads(report.get('issues_list', '[]'))
                report['time_metadata'] = json.loads(report.get('time_metadata', '{}'))
                
                return report
            
            return None
            
        except Exception as e:
            st.error(f"Error fetching health report: {str(e)}")
            return None
    
    def needs_health_report_update(self, project_id: str) -> bool:
        """Check if health report needs to be regenerated"""
        try:
            # Get latest report
            report = self.get_health_report(project_id)
            
            if not report:
                return True  # No report exists
            
            # Get current parameters
            parameters = self.get_project_parameters(project_id)
            current_params = sorted([p['parameter_name'] for p in parameters])
            
            # Get parameters from report
            report_params = sorted(report.get('parameters_analyzed', []))
            
            # Check if parameters have changed
            if current_params != report_params:
                return True  # New parameters added or removed
            
            # Check if data hash is different
            import hashlib
            param_string = json.dumps(current_params)
            current_hash = hashlib.md5(param_string.encode()).hexdigest()
            
            if current_hash != report.get('data_hash'):
                return True  # Data has changed
            
            return False  # Report is up to date
            
        except Exception as e:
            st.error(f"Error checking report status: {str(e)}")
            return True  # On error, assume update needed
    
    def generate_health_report_from_parameters(self, project_id: str) -> Dict[str, Any]:
        """Generate a complete health report from database parameters"""
        try:
            parameters = self.get_project_parameters(project_id)
            
            if not parameters:
                return {
                    'success': False,
                    'message': 'No parameters to analyze'
                }
            
            # Calculate health metrics
            health_score = 100
            issues = []
            critical_issues = 0
            warnings = 0
            
            # Missing values analysis
            missing_values_detail = {}
            total_missing = 0
            total_count = 0
            
            for param in parameters:
                param_name = param['parameter_name']
                missing_count = param.get('missing_count', 0)
                param_total = param.get('total_count', 0)
                
                total_missing += missing_count
                total_count += param_total
                
                if param_total > 0:
                    missing_pct = missing_count / param_total
                    
                    missing_values_detail[param_name] = {
                        'count': missing_count,
                        'percentage': missing_pct,
                        'total': param_total
                    }
                    
                    if missing_pct > 0.20:
                        health_score -= 15
                        issues.append(f"⚠️ {param_name}: {missing_pct*100:.1f}% missing (critical)")
                        critical_issues += 1
                    elif missing_pct > 0.05:
                        health_score -= 5
                        issues.append(f"⚠️ {param_name}: {missing_pct*100:.1f}% missing")
                        warnings += 1
            
            # Calculate overall metrics
            overall_missing_pct = total_missing / total_count if total_count > 0 else 0
            
            # Determine category
            health_score = max(0, min(100, health_score))
            
            if health_score >= 85:
                category = "excellent"
            elif health_score >= 70:
                category = "good"
            elif health_score >= 50:
                category = "fair"
            else:
                category = "poor"
            
            # Build health report
            health_report = {
                'success': True,
                'health_score': health_score,
                'health_category': category,
                'total_parameters': len(parameters),
                'total_data_points': total_count,
                'total_missing_values': total_missing,
                'missing_percentage': overall_missing_pct,
                'critical_issues': critical_issues,
                'warnings': warnings,
                'duplicate_timestamps': 0,  # Would need time series data to calculate
                'outlier_count': 0,  # Would need raw data to calculate
                'missing_values_detail': missing_values_detail,
                'outliers_detail': {},
                'coverage_detail': {},
                'issues_list': issues,
                'time_metadata': {},
                'parameters_analyzed': [p['parameter_name'] for p in parameters]
            }
            
            # Save to database
            self.save_health_report(project_id, health_report)
            
            return health_report
            
        except Exception as e:
            st.error(f"Error generating health report: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }


# Singleton pattern with caching
@st.cache_resource
def get_db_manager() -> SupabaseManager:
    """Get cached database manager instance"""
    return SupabaseManager()
