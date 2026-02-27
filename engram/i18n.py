"""
Internationalization (i18n) support for Engram.

This module provides translation capabilities for user-facing strings.
Usage:
    from engram.i18n import _, set_language
    
    # Set language (default: 'en')
    set_language('zh_CN')
    
    # Translate a string
    print(_("Hello, World!"))
"""

import gettext
import os
from typing import Optional

# Default language
_DEFAULT_LANGUAGE = 'en'
_CURRENT_LANGUAGE = _DEFAULT_LANGUAGE

# Translation object (initialized lazily)
_translation = None

def get_locales_dir() -> str:
    """Get the directory containing locale files."""
    return os.path.join(os.path.dirname(__file__), 'locales')

def set_language(lang_code: str) -> None:
    """
    Set the current language for translations.
    
    Args:
        lang_code: Language code (e.g., 'en', 'zh_CN')
    """
    global _CURRENT_LANGUAGE, _translation
    
    _CURRENT_LANGUAGE = lang_code
    
    locales_dir = get_locales_dir()
    
    try:
        _translation = gettext.translation(
            'engram',
            localedir=locales_dir,
            languages=[lang_code],
            fallback=(lang_code == _DEFAULT_LANGUAGE)
        )
    except FileNotFoundError:
        # Fallback to English if translation file not found
        _translation = gettext.NullTranslations()

def _(message: str) -> str:
    """
    Translate a message.
    
    Args:
        message: The message to translate
        
    Returns:
        The translated message
    """
    global _translation
    
    if _translation is None:
        set_language(_DEFAULT_LANGUAGE)
    
    return _translation.gettext(message)

def get_current_language() -> str:
    """Get the currently set language code."""
    return _CURRENT_LANGUAGE

# Initialize with default language on module import
set_language(_DEFAULT_LANGUAGE)
