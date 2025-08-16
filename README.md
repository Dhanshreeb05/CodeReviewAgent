# CodeReviewAgent

A project to automate code review feedback and patch classification using machine learning and natural language processing.

## Project Structure

- **codeReviewer/**  
  FastAPI service for automated code review classification (e.g., "Review Needed" vs "No Review Needed").
- **commentGeneration/**  
  FastAPI service for generating review comments for code diffs.
- **streamlit-ui/**  
  Streamlit web UI for interacting with the code review and comment generation services.
- **trainCodeReviewModel.py**  
  Python file for model training, evaluation, and analysis.
- **models/**  
  Contains trained model weights, tokenizer files, and configs.