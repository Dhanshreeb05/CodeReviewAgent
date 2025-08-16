import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
import json

from demo import (
    CodeReviewClassifier, 
    format_diff_for_model,
    preprocess_single_patch,
    predict_code_review,
    generate_review_comment
)

# ============== MODEL TESTS ==============
class TestCodeReviewClassifier:
    """Test the main classifier model"""
    
    def test_model_initialization(self):
        """Test model creates successfully"""
        model = CodeReviewClassifier()
        assert isinstance(model, nn.Module)
        assert model.codebert_dim == 768
        
    def test_forward_pass(self):
        """Test model forward pass works"""
        model = CodeReviewClassifier()
        
        # Creating dummy inputs
        input_ids = torch.randint(0, 1000, (2, 128))
        attention_mask = torch.ones(2, 128)
        numerical_features = torch.randn(2, 8)
        
        
        output = model(input_ids, attention_mask, numerical_features)
        assert output.shape == (2, 2)  # batch_size=2, num_classes=2


# ============== PREPROCESSING TESTS ==============
class TestPreprocessing:
    """Test preprocessing functions"""
    
    def test_format_diff_basic(self):
        """Test basic diff formatting"""
        diff = """+ added line
- removed line
  unchanged line"""
        
        result = format_diff_for_model(diff)
        assert "[ADD]" in result
        assert "[DEL]" in result
        assert "[KEEP]" in result
        
    def test_format_diff_empty(self):
        """Test empty diff handling"""
        assert format_diff_for_model("") == ""
        assert format_diff_for_model(None) == ""
        
    def test_preprocess_patch_python(self):
        """Test preprocessing Python patches"""
        patch = """@@ -10,7 +10,7 @@
+ added line
- removed line"""
        
        features = preprocess_single_patch(patch, lang="py")
        
        # Checking whether key features exist
        assert features['is_python'] == 1
        assert features['patch_length'] > 0
        assert features['num_additions'] >= 0
        assert features['num_deletions'] >= 0
        
    def test_preprocess_patch_no_message(self):
        """Test patch without commit message"""
        features = preprocess_single_patch("+ test", msg_text="")
        
        assert features['has_message'] == 0
        assert features['message_length'] == 0



# ============== EDGE CASES ==============
class TestEdgeCases:
    """Test edge cases"""
    
    def test_empty_patch(self):
        """Test empty patch handling"""
        features = preprocess_single_patch("")
        assert features['patch_length'] == 0
        assert features['total_changes'] == 0
        
    def test_unicode_patch(self):
        """Test unicode in patches"""
        unicode_patch = "+ 中文注释\n- English"
        result = format_diff_for_model(unicode_patch)
        assert isinstance(result, str)
        assert "[ADD]" in result
        
    def test_malformed_diff(self):
        """Test malformed diff doesn't crash"""
        bad_diff = "This is not a diff"
        result = format_diff_for_model(bad_diff)
        assert isinstance(result, str)  # Should handle gracefully


# ============== INTEGRATION TESTS ==============
@pytest.mark.parametrize("patch,expected_complexity", [
    # no review needed change
    ("+ import logging", "low"),
    ("+ MAX_RETRIES = 3", "low"),
    
    #  need review type changes 
    ("- return a / b\n+ return a / b if b != 0 else None", "high"),
    ("- query = f\"SELECT * WHERE user='{user}'\"\n+ query = \"SELECT * WHERE user=%s\"", "high"),
])
def test_patch_complexity_detection(patch, expected_complexity):
    """Test if different patch types are categorized correctly"""
    features = preprocess_single_patch(patch)
    
    if expected_complexity == "high":
        assert features['total_changes'] > 0
    else:
        assert features['num_deletions'] == 0 or features['total_changes'] <= 2


# ============== FIXTURES ==============
@pytest.fixture
def sample_patches():
    """Provide sample patches for testing"""
    return [
        """@@ -5,2 +5,2 @@
-def calculate(price, discount):
+def calculate(price: float, discount: float) -> float:
    return price * (1 - discount)""",
        
        """@@ -44,3 +44,3 @@
def divide(a, b):
-    return a / b
+    return a / b if b != 0 else None""",
    ]


@pytest.fixture
def mock_model():
    """Providing a mocked model for testing"""
    model = Mock()
    model.parameters.return_value = [torch.tensor([0.0])]
    model.eval = Mock()
    return model


