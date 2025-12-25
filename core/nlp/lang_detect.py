"""
Language detection module
"""

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent results
DetectorFactory.seed = 0


def detect_language(text: str) -> dict:
    """
    Detect language of input text
    
    Parameters:
    -----------
    text : str
        Input text to analyze
        
    Returns:
    --------
    result : dict
        {
            'detected_language': str (ISO code),
            'is_english': bool,
            'confidence': float (0-1)
        }
    """
    if not text or not text.strip():
        return {
            'detected_language': 'unknown',
            'is_english': False,
            'confidence': 0.0
        }
    
    try:
        # Detect language
        lang_code = detect(text)
        
        # Check if English
        is_english = lang_code == 'en'
        
        # For langdetect, we don't get confidence directly
        # but we can infer high confidence for clear cases
        # For v1, we'll use a simple heuristic
        confidence = 0.95 if is_english else 0.85
        
        return {
            'detected_language': lang_code,
            'is_english': is_english,
            'confidence': confidence
        }
    
    except LangDetectException:
        # If detection fails, assume English but with low confidence
        return {
            'detected_language': 'unknown',
            'is_english': True,  # Permissive for v1
            'confidence': 0.3
        }


def validate_english(text: str, min_confidence: float = 0.5) -> bool:
    """
    Validate if text is English with sufficient confidence
    
    Parameters:
    -----------
    text : str
        Input text
    min_confidence : float
        Minimum confidence threshold (default: 0.5)
        
    Returns:
    --------
    is_valid : bool
        True if text is English with sufficient confidence
    """
    result = detect_language(text)
    return result['is_english'] and result['confidence'] >= min_confidence
