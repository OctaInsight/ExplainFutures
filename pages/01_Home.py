"""
Page 1: User Dashboard (Updated)
Project selection and management with multi-user support
"""

import streamlit as st
from datetime import datetime
import time
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Dashboard - ExplainFutures",
    page_icon=str(Path("assets/logo_small.png")),
    layout="wide"
)

# Import database manager
try:
    from core.database.supabase_manager import get_db_manager
    from core.shared_sidebar import render_app_sidebar
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    st.error(f"⚠️ Import error: {str(e)}")


def check_authentication():
    """Check if user is authenticated, redirect to login if not"""
    if not st.session_state.get('authenticated', False):
        st.warning("⚠️ Please log in to continue")
        time.sleep(1)
        st.switch_page("App.py")  # Redirect to App.py (login is there)
        st.stop()


def show_demo_timer():
    """Show countdown timer for demo users"""
    if st.session_state.get('is_demo') and st.session_state.get('demo_expires_at'):
        expires_at = datetime.fromisoformat(st.session_state.demo_expires_at.replace('Z', '+00:00'))
        remaining = (expires_at - datetime.now()).total_seconds()
        
        if remaining > 0:
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            
            st.sidebar.warning(f"🎭 Demo expires in: {minutes:02d}:{seconds:02d}")
        else:
            st.sidebar.error("🎭 Demo session expired!")
            if st.sidebar.button("Start New Demo"):
                logout_user()


def logout_user():
    """Logout and cleanup"""
    db = get_db_manager()
    
    # Cleanup demo session if needed
    if st.session_state.get('is_demo') and st.session_state.get('demo_session_id'):
        try:
            db.end_demo_session(st.session_state.demo_session_id)
            st.success("🎭 Demo session cleaned up!")
        except Exception as e:
            st.error(f"Cleanup error: {str(e)}")
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.success("✅ Logged out successfully")
    time.sleep(1)
    st.switch_page("App.py")  # Back to login page


def create_new_project():
    """Show create project dialog"""
    st.subheader("📁 Create New Project")
    
    with st.form("create_project_form"):
        project_name = st.text_input(
            "Project Name*",
            placeholder="e.g., Climate Scenarios 2050",
            help="Give your project a descriptive name"
        )
        
        description = st.text_area(
            "Description",
            placeholder="Brief description of your project goals...",
            height=100
        )
        
        submitted = st.form_submit_button("Create Project", type="primary", use_container_width=True)
        
        if submitted:
            if not project_name:
                st.error("⚠️ Project name is required")
                return
            
            db = get_db_manager()
            
            # Check limits
            limits = db.check_user_limits(st.session_state.user_id)
            
            if not limits['can_create_project']:
                st.error("❌ You've reached your project limit. Please upgrade or delete old projects.")
                return
            
            # Create project
            try:
                project = db.create_project(
                    owner_id=st.session_state.user_id,
                    project_name=project_name,
                    description=description
                )
                
                if project:
                    st.success(f"✅ Project created: {project['project_code']}")
                    st.session_state.current_project_id = project['project_id']
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to create project")
                    
            except Exception as e:
                st.error(f"Error creating project: {str(e)}")


def show_user_stats(db):
    """Show user statistics and limits"""
    # FIXED: Use db.client instead of db.admin_client
    user = db.client.table('users').select('*').eq('user_id', st.session_state.user_id).execute()
    
    if user.data:
        user_data = user.data[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Projects",
                f"{user_data.get('current_project_count', 0)} / {user_data.get('max_projects', 3)}",
                help="Number of active projects"
            )
        
        with col2:
            st.metric(
                "Storage",
                f"{user_data.get('current_storage_mb', 0):.1f} / {user_data.get('max_storage_mb', 1000)} MB",
                help="Storage space used"
            )
        
        with col3:
            st.metric(
                "Uploads This Month",
                f"{user_data.get('uploads_this_month', 0)} / {user_data.get('max_uploads_per_month', 50)}",
                help="Files uploaded this month"
            )
        
        with col4:
            tier = user_data.get('subscription_tier', 'free').upper()
            st.metric(
                "Subscription",
                tier,
                help="Current subscription tier"
            )


def show_projects_list(db):
    """Show user's projects"""
    st.subheader("📂 My Projects")
    
    # Get projects
    projects = db.get_user_projects(
        user_id=st.session_state.user_id,
        include_collaborations=True
    )
    
    if not projects:
        st.info("🎯 No projects yet. Create your first project to get started!")
        return
    
    # Debug: Show project count
    st.caption(f"Total projects loaded: {len(projects)}")
    
    # Separate owned and shared projects
    owned_projects = [p for p in projects if p.get('is_owner', True)]
    shared_projects = [p for p in projects if not p.get('is_owner', True)]
    
    # Debug: Show counts
    st.caption(f"Owned: {len(owned_projects)} | Shared with me: {len(shared_projects)}")
    
    # Create tabs for owned and shared
    if shared_projects:
        tab_owned, tab_shared = st.tabs([
            f"📁 My Projects ({len(owned_projects)})",
            f"👥 Shared with Me ({len(shared_projects)})"
        ])
    else:
        tab_owned = st.container()
        tab_shared = None
    
    # Show owned projects
    with tab_owned:
        show_project_section(db, owned_projects, "owned")
    
    # Show shared projects if any
    if tab_shared:
        with tab_shared:
            show_project_section(db, shared_projects, "shared")


def show_project_section(db, projects, section_type):
    """Show a section of projects (owned or shared)"""
    if not projects:
        if section_type == "shared":
            st.info("📭 No projects have been shared with you yet.")
        else:
            st.info("🎯 No projects yet. Create your first project to get started!")
        return
    
    # Filter options
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        filter_status = st.selectbox(
            "Status",
            options=["All", "Active", "Archived"],
            index=0,
            key=f"status_{section_type}"
        )
    
    with col_filter2:
        sort_by = st.selectbox(
            "Sort by",
            options=["Last Accessed", "Created Date", "Name", "Progress"],
            index=0,
            key=f"sort_{section_type}"
        )
    
    with col_filter3:
        view_mode = st.radio(
            "View",
            options=["Grid", "List"],
            index=0,
            horizontal=True,
            key=f"view_{section_type}"
        )
    
    # Filter projects
    filtered_projects = projects
    
    if filter_status == "Active":
        filtered_projects = [p for p in projects if p['status'] == 'active']
    elif filter_status == "Archived":
        filtered_projects = [p for p in projects if p['status'] == 'archived']
    
    # Sort projects
    if sort_by == "Last Accessed":
        filtered_projects.sort(key=lambda x: x.get('last_accessed', ''), reverse=True)
    elif sort_by == "Created Date":
        filtered_projects.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    elif sort_by == "Name":
        filtered_projects.sort(key=lambda x: x.get('project_name', ''))
    elif sort_by == "Progress":
        filtered_projects.sort(key=lambda x: x.get('completion_percentage', 0), reverse=True)
    
    st.markdown("---")
    
    # Display projects
    if view_mode == "Grid":
        # Grid view (3 columns)
        cols = st.columns(3)
        
        for idx, project in enumerate(filtered_projects):
            with cols[idx % 3]:
                show_project_card(project, db)
    
    else:
        # List view
        for project in filtered_projects:
            show_project_row(project, db)


def show_project_card(project, db):
    """Show project as card (grid view)"""
    is_owner = project.get('is_owner', project['owner_id'] == st.session_state.user_id)
    is_demo = project.get('is_demo_project', False)
    
    with st.container(border=True):
        # Header
        st.markdown(f"**{project['project_name']}**")
        
        if is_demo:
            st.caption("🎭 Demo Project")
        elif is_owner:
            st.caption("👤 Owner")
        else:
            access_role = project.get('access_role', 'collaborator')
            st.caption(f"👥 {access_role.title()}")
        
        # Show collaborators if owner
        if is_owner and not is_demo:
            collaborators = db.get_project_collaborators(project['project_id'])
            if collaborators:
                collab_names = ", ".join([c['username'] for c in collaborators[:2]])
                if len(collaborators) > 2:
                    collab_names += f" +{len(collaborators) - 2} more"
                st.caption(f"🤝 Shared with: {collab_names}")
        
        # Progress
        progress = project.get('completion_percentage', 0)
        st.progress(progress / 100, text=f"Progress: {progress}%")
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"📄 {project.get('total_parameters', 0)} params")
        with col2:
            st.caption(f"📊 {project.get('total_scenarios', 0)} scenarios")
        
        # Actions
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Open", key=f"open_{project['project_id']}", use_container_width=True):
                st.session_state.current_project_id = project['project_id']
                
                # Update last accessed
                db.update_project_progress(
                    project_id=project['project_id'],
                    current_page=project.get('current_page', 2)
                )
                
                # Navigate to Data Import page
                st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
        
        with col_btn2:
            if st.button("⋯", key=f"menu_{project['project_id']}", use_container_width=True):
                st.session_state[f"show_menu_{project['project_id']}"] = not st.session_state.get(f"show_menu_{project['project_id']}", False)
                st.rerun()
        
        # Show menu options if toggled
        if st.session_state.get(f"show_menu_{project['project_id']}", False):
            st.markdown("---")
            
            if is_owner:
                col_menu1, col_menu2 = st.columns(2)
                
                with col_menu1:
                    if st.button("Share", key=f"share_card_{project['project_id']}", use_container_width=True):
                        st.session_state[f"show_share_{project['project_id']}"] = True
                        st.session_state[f"show_menu_{project['project_id']}"] = False
                        st.rerun()
                
                with col_menu2:
                    if not is_demo:
                        if st.button("Delete", key=f"delete_card_{project['project_id']}", use_container_width=True):
                            if st.session_state.get(f"confirm_delete_{project['project_id']}"):
                                db.delete_project(project['project_id'], st.session_state.user_id)
                                st.success("Project deleted")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.session_state[f"confirm_delete_{project['project_id']}"] = True
                                st.warning("⚠️ Click again to confirm")
                
                # Show collaborators management
                if st.button("Manage Access", key=f"manage_collab_card_{project['project_id']}", use_container_width=True):
                    st.session_state[f"show_collaborators_{project['project_id']}"] = True
                    st.session_state[f"show_menu_{project['project_id']}"] = False
                    st.rerun()
            else:
                # For collaborators, show limited options
                st.caption(f"Access: {project.get('access_role', 'collaborator').title()}")
                if project.get('can_edit'):
                    st.caption("✓ Can edit")
        
        # Collaborators management dialog
        if st.session_state.get(f"show_collaborators_{project['project_id']}", False):
            st.markdown("---")
            st.markdown("**👥 Project Access**")
            
            collaborators = db.get_project_collaborators(project['project_id'])
            
            if collaborators:
                for collab in collaborators:
                    col_name, col_remove = st.columns([3, 1])
                    with col_name:
                        st.text(f"{collab['full_name']} ({collab['username']})")
                        st.caption(f"Role: {collab['role']}")
                    with col_remove:
                        if st.button("Remove", key=f"remove_{collab['collaborator_id']}"):
                            if db.remove_collaborator(project['project_id'], collab['user_id'], st.session_state.user_id):
                                st.success(f"Removed {collab['username']}")
                                time.sleep(1)
                                st.rerun()
            else:
                st.caption("No collaborators yet")
            
            if st.button("Close", key=f"close_collab_card_{project['project_id']}", use_container_width=True):
                st.session_state[f"show_collaborators_{project['project_id']}"] = False
                st.rerun()
        
        # Share dialog (same as before)
        if st.session_state.get(f"show_share_{project['project_id']}", False):
            st.markdown("---")
            st.markdown("**👥 Share Project**")
            
            with st.form(key=f"share_form_card_{project['project_id']}"):
                share_email = st.text_input(
                    "Email",
                    placeholder="user@example.com",
                    label_visibility="collapsed"
                )
                
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    share_btn = st.form_submit_button("Share", type="primary", use_container_width=True)
                
                with col_s2:
                    cancel_btn = st.form_submit_button("Cancel", use_container_width=True)
                
                if share_btn:
                    if not share_email:
                        st.error("⚠️ Enter email")
                    else:
                        user_result = db.client.table('users').select('user_id, username, email').eq('email', share_email).execute()
                        
                        if user_result.data:
                            target_user = user_result.data[0]
                            existing = db.client.table('project_collaborators').select('*').eq('project_id', project['project_id']).eq('user_id', target_user['user_id']).execute()
                            
                            if existing.data:
                                st.warning(f"Already shared with {target_user['username']}")
                            else:
                                try:
                                    db.client.table('project_collaborators').insert({
                                        'project_id': project['project_id'],
                                        'user_id': target_user['user_id'],
                                        'role': 'collaborator',
                                        'can_edit': True,
                                        'can_delete': False
                                    }).execute()
                                    
                                    st.success(f"✅ Shared with {target_user['username']}")
                                    time.sleep(2)
                                    st.session_state[f"show_share_{project['project_id']}"] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        else:
                            st.warning(f"No user: {share_email}")
                            st.info("Ask them to register first")
                
                if cancel_btn:
                    st.session_state[f"show_share_{project['project_id']}"] = False
                    st.rerun()


def show_project_row(project, db):
    """Show project as row (list view)"""
    is_owner = project.get('is_owner', project['owner_id'] == st.session_state.user_id)
    is_demo = project.get('is_demo_project', False)
    
    with st.expander(f"{'🎭' if is_demo else '📁'} {project['project_name']}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Description:** {project.get('description', 'No description')}")
            st.caption(f"**Code:** {project.get('project_code', 'N/A')}")
            st.caption(f"**Created:** {datetime.fromisoformat(project['created_at'].replace('Z', '')).strftime('%Y-%m-%d')}")
            
            if not is_owner:
                st.caption(f"👥 Shared by owner • Role: {project.get('access_role', 'collaborator').title()}")
            else:
                # Show collaborators for owner
                collaborators = db.get_project_collaborators(project['project_id'])
                if collaborators:
                    collab_list = ", ".join([c['username'] for c in collaborators])
                    st.caption(f"🤝 **Collaborators:** {collab_list}")
        
        with col2:
            st.metric("Progress", f"{project.get('completion_percentage', 0)}%")
            st.metric("Parameters", project.get('total_parameters', 0))
            st.metric("Scenarios", project.get('total_scenarios', 0))
        
        # Actions
        if is_owner:
            col_act1, col_act2, col_act3, col_act4 = st.columns(4)
        else:
            col_act1, col_act2 = st.columns([1, 3])
            col_act3 = col_act4 = None
        
        with col_act1:
            if st.button("Open Project", key=f"open_list_{project['project_id']}", type="primary"):
                st.session_state.current_project_id = project['project_id']
                db.update_project_progress(
                    project_id=project['project_id'],
                    current_page=project.get('current_page', 2)
                )
                # Navigate to Data Import page
                st.switch_page("pages/02_Data_Import_&_Diagnostics.py")
        
        if is_owner:
            with col_act2:
                if st.button("Manage Access", key=f"manage_access_{project['project_id']}"):
                    st.session_state[f"show_manage_access_{project['project_id']}"] = True
                    st.rerun()
            
            with col_act3:
                if st.button("Share", key=f"share_{project['project_id']}"):
                    st.session_state[f"show_share_{project['project_id']}"] = True
                    st.rerun()
            
            with col_act4:
                if not is_demo:
                    if st.button("Delete", key=f"delete_{project['project_id']}"):
                        if st.session_state.get(f"confirm_delete_{project['project_id']}"):
                            db.delete_project(project['project_id'], st.session_state.user_id)
                            st.success("Project deleted")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.session_state[f"confirm_delete_{project['project_id']}"] = True
                            st.warning("⚠️ Click again to confirm deletion")
        
        # Manage Access dialog
        if st.session_state.get(f"show_manage_access_{project['project_id']}", False):
            st.markdown("---")
            st.markdown("#### 👥 Manage Project Access")
            
            collaborators = db.get_project_collaborators(project['project_id'])
            
            if collaborators:
                for collab in collaborators:
                    col_user, col_role, col_actions = st.columns([2, 1, 1])
                    
                    with col_user:
                        st.markdown(f"**{collab['full_name']}**")
                        st.caption(collab['email'])
                    
                    with col_role:
                        st.text(collab['role'].title())
                        if collab['can_edit']:
                            st.caption("✓ Can edit")
                    
                    with col_actions:
                        if st.button("Remove", key=f"remove_list_{collab['collaborator_id']}"):
                            if db.remove_collaborator(project['project_id'], collab['user_id'], st.session_state.user_id):
                                st.success(f"Removed {collab['username']}")
                                time.sleep(1)
                                st.rerun()
            else:
                st.info("No collaborators yet. Use the 'Share' button to add collaborators.")
            
            if st.button("Close", key=f"close_manage_{project['project_id']}", use_container_width=True):
                st.session_state[f"show_manage_access_{project['project_id']}"] = False
                st.rerun()
        
        # Share dialog
        if st.session_state.get(f"show_share_{project['project_id']}", False):
            st.markdown("---")
            st.markdown("#### 👥 Share Project")
            
            with st.form(key=f"share_form_{project['project_id']}"):
                share_email = st.text_input(
                    "Enter email address",
                    placeholder="user@example.com",
                    help="Share this project with another user"
                )
                
                col_share1, col_share2 = st.columns(2)
                
                with col_share1:
                    share_button = st.form_submit_button("Share", type="primary", use_container_width=True)
                
                with col_share2:
                    cancel_share = st.form_submit_button("Cancel", use_container_width=True)
                
                if share_button:
                    if not share_email:
                        st.error("⚠️ Please enter an email address")
                    else:
                        # Check if user exists
                        user_result = db.client.table('users').select('user_id, username, email').eq('email', share_email).execute()
                        
                        if user_result.data:
                            # User exists - share the project
                            target_user = user_result.data[0]
                            
                            # Check if already shared
                            existing = db.client.table('project_collaborators').select('*').eq('project_id', project['project_id']).eq('user_id', target_user['user_id']).execute()
                            
                            if existing.data:
                                st.warning(f"⚠️ Project already shared with {target_user['username']}")
                            else:
                                # Add collaborator
                                try:
                                    db.client.table('project_collaborators').insert({
                                        'project_id': project['project_id'],
                                        'user_id': target_user['user_id'],
                                        'role': 'collaborator',
                                        'can_edit': True,
                                        'can_delete': False
                                    }).execute()
                                    
                                    st.success(f"✅ Project shared with {target_user['username']} ({share_email})")
                                    time.sleep(2)
                                    st.session_state[f"show_share_{project['project_id']}"] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error sharing project: {str(e)}")
                        else:
                            # User doesn't exist
                            st.warning(f"⚠️ No user found with email: {share_email}")
                            st.info("💡 Please ask them to register at the ExplainFutures platform first, then try sharing again.")
                
                if cancel_share:
                    st.session_state[f"show_share_{project['project_id']}"] = False
                    st.rerun()


def main():
    """Main dashboard"""
    
    check_authentication()
    
    # Sidebar
    try:
        render_app_sidebar()
    except:
        # Minimal sidebar
        st.sidebar.title("ExplainFutures")
        st.sidebar.markdown(f"**User:** {st.session_state.get('username', 'Unknown')}")
        
        if st.sidebar.button("Logout", use_container_width=True):
            logout_user()
    
    # Show demo timer if demo user
    if st.session_state.get('is_demo'):
        show_demo_timer()
    
    # Header
    st.title("🏠 Dashboard")
    st.markdown(f"**Welcome back, {st.session_state.get('full_name', 'User')}!**")
    
    if st.session_state.get('is_demo'):
        st.info("🎭 **Demo Mode**: Explore with pre-loaded data. Your session will expire in 30 minutes and changes will be reset.")
    
    st.markdown("---")
    
    # Get database
    db = get_db_manager()
    
    # User stats
    show_user_stats(db)
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📂 Projects", "➕ Create Project", "📊 Activity"])
    
    with tab1:
        show_projects_list(db)
    
    with tab2:
        if st.session_state.get('is_demo'):
            st.warning("🎭 Demo users cannot create projects. Use the pre-loaded demo project.")
        else:
            create_new_project()
    
    with tab3:
        st.info("📊 Activity feed coming soon")
        st.caption("Recent actions, collaborations, and notifications will appear here.")


if __name__ == "__main__":
    if not DB_AVAILABLE:
        st.stop()
    
    main()
