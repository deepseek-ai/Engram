"""
Engram - Conditional Memory via Scalable Lookup

This package provides the Engram module implementation for efficient
N-gram memory retrieval in transformer models.

Internationalization (i18n) is supported. To use a different language:
    from engram import set_language
    set_language('zh_CN')  # 设置为中文
"""

from engram.i18n import _, set_language, get_current_language, get_locales_dir

__version__ = "0.1.0"
__all__ = [
    '_',
    'set_language',
    'get_current_language', 
    'get_locales_dir',
]
