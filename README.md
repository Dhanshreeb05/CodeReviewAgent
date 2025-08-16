# CodeReviewAgent 🔍

An automated code review system that generates review comments and estimates code quality using transformer-based models.

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


## Quick Start

### Option 1: Try Online Demo
Visit our deployed web interface: **[https://streamlit-ui-586748801796.us-central1.run.app/](https://streamlit-ui-586748801796.us-central1.run.app/)**

### Option 2: Run Locally

#### Prerequisites
- Python 3.10+
- Git

#### Installation
```bash
# Clone the repository
git clone https://github.com/Dhanshreeb05/CodeReviewAgent.git
cd CodeReviewAgent

# Install dependencies
pip install -r requirements.txt
```

#### Usage
```bash
# Run the demo
python demo.py

# Run tests
pytest test_demo.py -v
```

## 📁 Project Structure
```
CodeReviewAgent/
├── demo.py                    # Main demo application
├── requirements.txt           # Python dependencies
├── commentGeneration/         # Comment generation models
│   └── models/
├── codeReviewer/             # Code quality models
│   └── models/
└── test_demo.py              # Unit tests
```

## 🔧 Configuration
Verify model paths in `demo.py`:
- **Line 274**: `COMMENT_GENERATION_MODEL_PATH = "commentGeneration/models"`
- **Line 334**: `CODE_REVIEWER_MODEL_PATH = "codeReviewer/models"`

## 💻 How to Use

### Web Interface
1. **Check API Status**: Ensure both Quality API and Comment API show "Online"
2. **Input Code Diff**: 
   - Select from pre-configured samples, OR
   - Paste your own git diff (follow UI instructions)
3. **Analyze**: Click "Analyze Code" button
4. **View Results**:
   - Quality estimation with confidence score
   - Generated review comment (if review needed)
   - Processing times and technical details

### Local Demo
Run `python demo.py` and follow the interactive prompts to:
- Input code diffs
- Get quality predictions
- Generate review comments

## 🎯 Features
- **Code Quality Estimation**: Binary prediction of whether code needs review
- **Comment Generation**: Automated review comment generation
- **Multi-language Support**: Works with various programming languages
- **Real-time Analysis**: Fast processing with confidence scores

## 📊 Models Used
- **CodeT5-Base**: For comment generation
- **Transformer-based**: For code quality estimation
- **Beam Search**: Decoding with beam size 5 for quality outputs

## 🧪 Testing
```bash
pytest test_demo.py -v
```
---

**Repository**: [https://github.com/Dhanshreeb05/CodeReviewAgent](https://github.com/Dhanshreeb05/CodeReviewAgent)
