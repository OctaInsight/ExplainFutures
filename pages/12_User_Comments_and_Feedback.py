"""
Page 12: User Comments and Feedback
Collect user feedback, feature requests, and suggestions with classification and export
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import sys

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Comments & Feedback",
    page_icon="💬",
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import initialize_session_state
from core.shared_sidebar import render_app_sidebar

# Initialize
initialize_session_state()

# Render shared sidebar
render_app_sidebar()

# === PAGE TITLE ===
st.title("💬 User Comments & Feedback")
st.markdown("*Help us improve the application by sharing your thoughts, suggestions, and feature requests*")
st.markdown("---")


def initialize_comments_state():
    """Initialize session state for comments"""
    if "user_comments" not in st.session_state:
        st.session_state.user_comments = []
    if "comment_counter" not in st.session_state:
        st.session_state.comment_counter = 0


def save_comment(name, email, category, comment_text, rating):
    """
    Save a comment to session state and optionally to file
    
    Parameters:
    -----------
    name : str
        User's name
    email : str
        User's email
    category : str
        Comment category
    comment_text : str
        The comment content
    rating : int
        User rating (1-5 stars)
    """
    
    st.session_state.comment_counter += 1
    
    comment_entry = {
        'id': st.session_state.comment_counter,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'name': name,
        'email': email,
        'category': category,
        'comment': comment_text,
        'rating': rating,
        'status': 'new'
    }
    
    st.session_state.user_comments.append(comment_entry)
    
    # Optionally save to file (for persistence across sessions)
    try:
        comments_file = Path('user_comments.json')
        
        # Load existing comments if file exists
        if comments_file.exists():
            with open(comments_file, 'r') as f:
                all_comments = json.load(f)
        else:
            all_comments = []
        
        # Add new comment
        all_comments.append(comment_entry)
        
        # Save back to file
        with open(comments_file, 'w') as f:
            json.dump(all_comments, f, indent=2)
        
        return True, "Comment saved successfully!"
    
    except Exception as e:
        return False, f"Comment saved to session but couldn't save to file: {str(e)}"


def export_comments_to_csv():
    """Export all comments to CSV format"""
    if not st.session_state.user_comments:
        return None
    
    df = pd.DataFrame(st.session_state.user_comments)
    return df.to_csv(index=False)


def export_comments_to_json():
    """Export all comments to JSON format"""
    if not st.session_state.user_comments:
        return None
    
    return json.dumps(st.session_state.user_comments, indent=2)


def main():
    """Main page function"""
    
    # Initialize state
    initialize_comments_state()
    
    # === INFORMATION PANEL ===
    with st.expander("ℹ️ About This Page", expanded=False):
        st.markdown("""
        ### Why Your Feedback Matters
        
        This is the **first version** of the application, and your input is invaluable for:
        
        - 🎨 **Improving the user interface and experience**
        - 🔬 **Enhancing scientific accuracy and methodology**
        - ✨ **Adding new features and capabilities**
        - 🐛 **Identifying and fixing bugs**
        - 📚 **Creating better documentation**
        
        **What You Can Do Here:**
        1. Submit feedback about any aspect of the application
        2. Request new features or improvements
        3. Report bugs or issues
        4. Share your scientific suggestions
        5. Rate your overall experience
        
        **Comment Categories:**
        - **🎨 Stylistic/UI**: Interface design, colors, layout, usability
        - **🔬 Scientific**: Methodology, models, algorithms, accuracy
        - **✨ Feature Request**: New features you'd like to see
        - **🐛 Bug Report**: Issues, errors, or unexpected behavior
        - **📚 Documentation**: Help text, tutorials, explanations
        - **⚡ Performance**: Speed, efficiency, optimization
        - **💡 General Feedback**: Other thoughts and suggestions
        
        All feedback is appreciated and will be reviewed!
        """)
    
    st.markdown("---")
    
    # === SUBMIT FEEDBACK SECTION ===
    st.subheader("📝 Submit Your Feedback")
    st.caption("Help us improve by sharing your thoughts and suggestions")
    
    # Check if user is logged in (check session state for user info)
    is_demo_user = st.session_state.get('is_demo_user', True)  # Default to demo mode
    user_name = st.session_state.get('user_name', '')
    user_email = st.session_state.get('user_email', '')
    
    with st.form("comment_form", clear_on_submit=True):
        # User Information - Only show for demo users
        if is_demo_user:
            st.markdown("### 👤 Your Information (Optional)")
            col1, col2 = st.columns(2)
            
            with col1:
                form_user_name = st.text_input(
                    "Your Name",
                    placeholder="e.g., Dr. Jane Smith",
                    help="Your name will be stored with your comment"
                )
            
            with col2:
                form_user_email = st.text_input(
                    "Email",
                    placeholder="your.email@example.com",
                    help="In case we need to follow up on your feedback"
                )
        else:
            # User is logged in - use their credentials
            form_user_name = user_name
            form_user_email = user_email
            
            # Show confirmation of logged-in user
            st.info(f"✅ Logged in as: **{user_name}** ({user_email})")
        
        # Comment Category
        st.markdown("### 📂 Category")
        
        comment_category = st.selectbox(
            "Select the type of feedback:",
            options=[
                "🎨 Stylistic/UI",
                "🔬 Scientific",
                "✨ Feature Request",
                "🐛 Bug Report",
                "📚 Documentation",
                "⚡ Performance",
                "💡 General Feedback"
            ],
            help="Choose the category that best fits your comment"
        )
        
        # Rating
        st.markdown("### ⭐ Overall Rating")
        
        rating = st.select_slider(
            "How would you rate your experience with this application?",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: "⭐" * x,
            help="1 = Poor, 5 = Excellent"
        )
        
        # Comment Text
        st.markdown("### 💬 Your Feedback")
        
        comment_text = st.text_area(
            "Share your thoughts, suggestions, or report issues:",
            height=200,
            placeholder="""Example:
- I love the forecasting feature, but the charts could use more color options.
- It would be great to export results in multiple formats.
- The dimensionality reduction page is very helpful for understanding complex datasets.
- I encountered an error when uploading a CSV with special characters.""",
            help="Be as detailed as possible - it helps us understand your feedback better"
        )
        
        # Submit button
        col_submit1, col_submit2, col_submit3 = st.columns([2, 1, 2])
        
        with col_submit2:
            submitted = st.form_submit_button(
                "📤 Submit Feedback",
                type="primary",
                use_container_width=True
            )
        
        if submitted:
            if comment_text.strip():
                # Determine final name and email
                final_name = form_user_name if form_user_name.strip() else "Anonymous"
                final_email = form_user_email if is_demo_user else user_email
                
                # Save comment
                success, message = save_comment(
                    name=final_name,
                    email=final_email,
                    category=comment_category,
                    comment_text=comment_text,
                    rating=rating
                )
                
                if success:
                    st.success("✅ Thank you for your feedback! Your comment has been submitted.")
                    
                    # Show summary
                    with st.expander("📋 Your Submitted Feedback", expanded=True):
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        
                        with col_sum1:
                            st.metric("Category", comment_category)
                        
                        with col_sum2:
                            st.metric("Rating", "⭐" * rating)
                        
                        with col_sum3:
                            st.metric("Timestamp", datetime.now().strftime('%H:%M:%S'))
                        
                        st.markdown("**Your Comment:**")
                        st.info(comment_text)
                        
                        st.caption("Your feedback has been recorded and will be reviewed by our team.")
                else:
                    st.warning(f"⚠️ {message}")
            else:
                st.error("❌ Please enter a comment before submitting.")
    
    # Show quick stats
    if st.session_state.user_comments:
        st.markdown("---")
        st.markdown("### 📊 Your Feedback History (This Session)")
        st.caption(f"You have submitted {len(st.session_state.user_comments)} comment(s) during this session")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Total Comments", len(st.session_state.user_comments))
        
        with col_stat2:
            avg_rating = sum(c['rating'] for c in st.session_state.user_comments) / len(st.session_state.user_comments)
            st.metric("Avg Rating", f"{'⭐' * int(avg_rating)} ({avg_rating:.1f})")
        
        with col_stat3:
            categories = [c['category'] for c in st.session_state.user_comments]
            most_common = max(set(categories), key=categories.count) if categories else "N/A"
            st.metric("Most Common", most_common.split()[1] if ' ' in most_common else most_common)
        
        with col_stat4:
            latest = st.session_state.user_comments[-1]['timestamp']
            st.metric("Latest", latest.split()[1])
        
        # Show recent comments summary
        st.markdown("**Recent Comments:**")
        for comment in reversed(st.session_state.user_comments[-3:]):  # Show last 3
            with st.container():
                col_rc1, col_rc2 = st.columns([1, 4])
                with col_rc1:
                    st.markdown(f"**{'⭐' * comment['rating']}**")
                with col_rc2:
                    st.caption(f"{comment['category']} • {comment['timestamp']}")
                    st.markdown(f"_{comment['comment'][:100]}..._" if len(comment['comment']) > 100 else f"_{comment['comment']}_")
                st.markdown("---")


if __name__ == "__main__":
    main()
