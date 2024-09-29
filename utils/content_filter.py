# utils/content_filter.py
def filter_inappropriate_content(text):
    """
    Filter and sanitize text to ensure it is age-appropriate.

    Args:
        text (str): The input text.

    Returns:
        str: The filtered and sanitized text.
    """
    # Implement basic content filtering logic
    inappropriate_words = ["badword1", "badword2"]  # Example list of words to filter
    for word in inappropriate_words:
        text = text.replace(word, "****")
    return text