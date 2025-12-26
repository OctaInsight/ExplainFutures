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
    
    # === CREATE TWO MAIN SECTIONS ===
    tab1, tab2 = st.tabs(["📝 Submit Feedback", "📋 Review Comments"])
    
    # === TAB 1: SUBMIT FEEDBACK ===
    with tab1:
        st.subheader("📝 Submit Your Feedback")
        st.caption("All fields are optional except the comment itself")
        
        with st.form("comment_form", clear_on_submit=True):
            # User Information
            col1, col2 = st.columns(2)
            
            with col1:
                user_name = st.text_input(
                    "Your Name (optional)",
                    placeholder="e.g., Dr. Jane Smith",
                    help="Your name will be stored with your comment"
                )
            
            with col2:
                user_email = st.text_input(
                    "Email (optional)",
                    placeholder="your.email@example.com",
                    help="In case we need to follow up on your feedback"
                )
            
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
                    # Save comment
                    success, message = save_comment(
                        name=user_name if user_name.strip() else "Anonymous",
                        email=user_email,
                        category=comment_category,
                        comment_text=comment_text,
                        rating=rating
                    )
                    
                    if success:
                        st.success("✅ Thank you for your feedback! Your comment has been submitted.")
                        st.balloons()
                        
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
                    else:
                        st.warning(f"⚠️ {message}")
                else:
                    st.error("❌ Please enter a comment before submitting.")
        
        # Show quick stats
        if st.session_state.user_comments:
            st.markdown("---")
            st.markdown("### 📊 Your Feedback Stats (This Session)")
            
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
    
    # === TAB 2: REVIEW COMMENTS ===
    with tab2:
        st.subheader("📋 Review All Feedback")
        
        if not st.session_state.user_comments:
            st.info("No comments yet. Be the first to share your feedback!")
        else:
            # Filter options
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                filter_category = st.selectbox(
                    "Filter by Category:",
                    options=["All Categories"] + [
                        "🎨 Stylistic/UI",
                        "🔬 Scientific",
                        "✨ Feature Request",
                        "🐛 Bug Report",
                        "📚 Documentation",
                        "⚡ Performance",
                        "💡 General Feedback"
                    ],
                    key="filter_cat"
                )
            
            with col_filter2:
                filter_rating = st.selectbox(
                    "Filter by Rating:",
                    options=["All Ratings", "⭐⭐⭐⭐⭐ (5 stars)", "⭐⭐⭐⭐ (4 stars)", 
                            "⭐⭐⭐ (3 stars)", "⭐⭐ (2 stars)", "⭐ (1 star)"],
                    key="filter_rating"
                )
            
            with col_filter3:
                sort_by = st.selectbox(
                    "Sort by:",
                    options=["Newest First", "Oldest First", "Highest Rating", "Lowest Rating"],
                    key="sort_by"
                )
            
            # Apply filters
            filtered_comments = st.session_state.user_comments.copy()
            
            # Category filter
            if filter_category != "All Categories":
                filtered_comments = [c for c in filtered_comments if c['category'] == filter_category]
            
            # Rating filter
            if filter_rating != "All Ratings":
                rating_value = filter_rating.count('⭐')
                filtered_comments = [c for c in filtered_comments if c['rating'] == rating_value]
            
            # Sort
            if sort_by == "Newest First":
                filtered_comments = sorted(filtered_comments, key=lambda x: x['timestamp'], reverse=True)
            elif sort_by == "Oldest First":
                filtered_comments = sorted(filtered_comments, key=lambda x: x['timestamp'])
            elif sort_by == "Highest Rating":
                filtered_comments = sorted(filtered_comments, key=lambda x: x['rating'], reverse=True)
            elif sort_by == "Lowest Rating":
                filtered_comments = sorted(filtered_comments, key=lambda x: x['rating'])
            
            # Display summary
            st.markdown(f"**Showing {len(filtered_comments)} of {len(st.session_state.user_comments)} comments**")
            
            st.markdown("---")
            
            # Display comments
            if filtered_comments:
                for idx, comment in enumerate(filtered_comments):
                    with st.expander(
                        f"{comment['category']} - {'⭐' * comment['rating']} - {comment['name']} - {comment['timestamp']}", 
                        expanded=False
                    ):
                        col_c1, col_c2, col_c3, col_c4 = st.columns([2, 2, 2, 1])
                        
                        with col_c1:
                            st.markdown(f"**👤 Name:** {comment['name']}")
                        
                        with col_c2:
                            if comment['email']:
                                st.markdown(f"**📧 Email:** {comment['email']}")
                            else:
                                st.markdown("**📧 Email:** *(not provided)*")
                        
                        with col_c3:
                            st.markdown(f"**📂 Category:** {comment['category']}")
                        
                        with col_c4:
                            st.markdown(f"**⭐ Rating:** {comment['rating']}/5")
                        
                        st.markdown("---")
                        st.markdown("**💬 Comment:**")
                        st.markdown(f"> {comment['comment']}")
                        
                        # Action buttons
                        col_act1, col_act2 = st.columns([4, 1])
                        
                        with col_act2:
                            if st.button("🗑️ Delete", key=f"del_{comment['id']}"):
                                st.session_state.user_comments.remove(comment)
                                st.rerun()
            else:
                st.info("No comments match the selected filters.")
            
            # Export options
            st.markdown("---")
            st.markdown("### 📥 Export Comments")
            
            col_export1, col_export2, col_export3 = st.columns(3)
            
            with col_export1:
                csv_data = export_comments_to_csv()
                if csv_data:
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_data,
                        file_name=f"user_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.button("📥 Download as CSV", disabled=True, use_container_width=True)
            
            with col_export2:
                json_data = export_comments_to_json()
                if json_data:
                    st.download_button(
                        label="📥 Download as JSON",
                        data=json_data,
                        file_name=f"user_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.button("📥 Download as JSON", disabled=True, use_container_width=True)
            
            with col_export3:
                if st.button("🗑️ Clear All Comments", use_container_width=True):
                    if st.session_state.user_comments:
                        st.session_state.user_comments = []
                        st.session_state.comment_counter = 0
                        st.success("✅ All comments cleared!")
                        st.rerun()
            
            # Statistics dashboard
            if st.session_state.user_comments:
                st.markdown("---")
                st.markdown("### 📊 Feedback Analytics")
                
                # Create DataFrame for analysis
                df_comments = pd.DataFrame(st.session_state.user_comments)
                
                col_analytics1, col_analytics2 = st.columns(2)
                
                with col_analytics1:
                    st.markdown("**Comments by Category**")
                    category_counts = df_comments['category'].value_counts()
                    st.bar_chart(category_counts)
                
                with col_analytics2:
                    st.markdown("**Ratings Distribution**")
                    rating_counts = df_comments['rating'].value_counts().sort_index()
                    st.bar_chart(rating_counts)


if __name__ == "__main__":
    main()
