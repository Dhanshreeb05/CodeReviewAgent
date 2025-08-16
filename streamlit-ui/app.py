import streamlit as st
import requests
import json
import time
import os
from datetime import datetime

# Configuration - Production URLs
QUALITY_API_URL = os.getenv("QUALITY_API_URL", "https://quality-service-586748801796.us-central1.run.app")
COMMENT_API_URL = os.getenv("COMMENT_API_URL", "https://comment-service-586748801796.us-central1.run.app")

def call_quality_api(code_diff: str) -> dict:
    """Call the Code Reviewer API"""
    try:
        response = requests.post(
            f"{QUALITY_API_URL}/predict-quality",
            json={"code_diff": code_diff},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Quality API Error: {str(e)}")
        return None

def call_comment_api(code_diff: str) -> dict:
    """Call the comment generation API"""
    try:
        response = requests.post(
            f"{COMMENT_API_URL}/generate-comment",
            json={"code_diff": code_diff},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Comment API Error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="AI Code Review Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Code Review Assistant")
    st.markdown("Get AI-powered code review analysis and comments")
    
    # Sidebar for API status
    with st.sidebar:
        st.header("🔧 Service Status")
        
        # Check Quality API
        try:
            response = requests.get(f"{QUALITY_API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Quality API: Online")
            else:
                st.error("❌ Quality API: Offline")
        except:
            st.error("❌ Quality API: Unreachable")
        
        # Check Comment API
        try:
            response = requests.get(f"{COMMENT_API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Comment API: Online")
            else:
                st.error("❌ Comment API: Offline")
        except:
            st.error("❌ Comment API: Unreachable")
    
    # Main input area
    st.header("📝 Input Code Diff")
    
    # Sample diffs for testing
    sample_diffs = {
        "Simple Change": """@@ -10,7 +10,7 @@ def hello():
        + import re
print("Hello, world!")""",
        
        "Variable Rename": """@@ -8,5 +8,5 @@
class User:
    def __init__(self, username, email):
        self.username = username
-        self.email = email
+        self.email_address = email""",
        
        "Security Fix": """@@ -15,7 +15,7 @@ def authenticate():
- query = f"SELECT * FROM users WHERE id={user_id}"
+ query = "SELECT * FROM users WHERE id=%s"
- cursor.execute(query)
+ cursor.execute(query, (user_id,))"""
    }
    
    # Sample selector
    selected_sample = st.selectbox("📋 Choose a sample diff:", [""] + list(sample_diffs.keys()))
    
    code_diff = st.text_area(
        "Paste your code diff here:",
        value=sample_diffs.get(selected_sample, ""),
        height=200,
        placeholder="Paste your git diff output here..."
    )
    
    # Analysis button
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze Code", type="primary")
    
    # Process analysis
    if analyze_button:
        if not code_diff.strip():
            st.warning("⚠️ Please enter a code diff to analyze")
        else:
            # Step 1: Quality Analysis
            st.header("📊 Analysis Results")
            
            with st.spinner("🔍 Analyzing code quality..."):
                start_time = time.time()
                quality_result = call_quality_api(code_diff)
                quality_time = time.time() - start_time
            
            if not quality_result:
                st.error("❌ Failed to get quality analysis")
                return
            
            # Display quality results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if quality_result["needs_review"]:
                    st.error("⚠️ **Code Needs Review**")
                    st.markdown(f"**Confidence:** {quality_result['confidence']:.1%}")
                    st.markdown(f"**Reasoning:** {quality_result['reasoning']}")
                else:
                    st.success("✅ **Code Looks Good**")
                    st.markdown(f"**Confidence:** {quality_result['confidence']:.1%}")
                    st.markdown(f"**Reasoning:** {quality_result['reasoning']}")
            
            with col2:
                st.metric("Quality Check", f"{quality_result['processing_time_ms']}ms")
            
            # Step 2: Comment Generation (if needed)
            if quality_result["needs_review"]:
                st.divider()
                
                with st.spinner("💬 Generating review comment..."):
                    comment_start = time.time()
                    comment_result = call_comment_api(code_diff)
                    comment_time = time.time() - comment_start
                
                if comment_result:
                    st.header("💬 Generated Review Comment")
                    
                    # Display generated comment
                    st.info(f"**💡 Suggested Comment:**\n\n{comment_result['generated_comment']}")
                    
                    # Comment details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Comment Confidence", f"{comment_result['confidence']:.1%}")
                    with col2:
                        st.metric("Generation Time", f"{comment_result['processing_time_ms']}ms")
                else:
                    st.error("❌ Failed to generate comment")
            else:
                st.info("💡 **No review comment needed** - Code appears ready to merge!")
            
            # Technical Details
            with st.expander("🔧 Technical Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Quality Analysis")
                    st.json({
                        "needs_review": quality_result["needs_review"],
                        "confidence": quality_result["confidence"],
                        "processing_time_ms": quality_result["processing_time_ms"],
                        "request_id": quality_result["request_id"][:8] + "..."
                    })
                
                with col2:
                    if quality_result["needs_review"] and 'comment_result' in locals():
                        st.subheader("Comment Generation")
                        st.json({
                            "generated_comment": comment_result["generated_comment"],
                            "confidence": comment_result["confidence"],
                            "processing_time_ms": comment_result["processing_time_ms"],
                            "request_id": comment_result["request_id"][:8] + "..."
                        })
                    else:
                        st.subheader("Comment Generation")
                        st.info("Skipped - No review needed")
    
    # Instructions
    st.header("📖 How to Use")
    
    with st.expander("Getting code diffs"):
        st.markdown("""
        **From Git:**
        ```bash
        git diff HEAD~1 HEAD > my_changes.diff
        ```
        
        **From GitHub Pull Request:**
        1. Go to your PR page
        2. Add `.diff` to the end of the URL
        3. Copy the diff content
        
        **Example:** `https://github.com/user/repo/pull/123.diff`
        """)
    
    with st.expander("Understanding Results"):
        st.markdown("""
        **🔍 Quality Analysis:**
        - **Needs Review**: Code has potential issues requiring human review
        - **Looks Good**: Code appears clean and follows best practices
        - **Confidence**: Model certainty about the prediction (0-100%)
        
        **💬 Comment Generation:**
        - **Generated Comment**: AI-suggested review comment
        - **Only shown if**: Quality check indicates review is needed
        - **Confidence**: Model certainty about the comment quality
        """)

if __name__ == "__main__":
    main()