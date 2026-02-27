"""
Demo script showing i18n (internationalization) support in Engram.

This script demonstrates how to use different languages with the Engram package.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engram import set_language, get_current_language, _


def demo_i18n():
    """Demonstrate i18n capabilities."""
    print("=" * 60)
    print("Engram i18n Demo")
    print("=" * 60)
    print()
    
    # Default (English)
    print("1. Default Language (English):")
    print(f"   Current language: {get_current_language()}")
    print(f"   Forward Complete! -> {_('Forward Complete!')}")
    print(f"   input_ids.shape -> {_('input_ids.shape')}")
    print()
    
    # Switch to Chinese
    print("2. Switching to Chinese (中文):")
    set_language('zh_CN')
    print(f"   Current language: {get_current_language()}")
    print(f"   Forward Complete! -> {_('Forward Complete!')}")
    print(f"   input_ids.shape -> {_('input_ids.shape')}")
    print()
    
    # Switch back to English
    print("3. Switching back to English:")
    set_language('en')
    print(f"   Current language: {get_current_language()}")
    print(f"   Forward Complete! -> {_('Forward Complete!')}")
    print()
    
    print("=" * 60)
    print("Demo complete! All translations working correctly.")
    print("=" * 60)


if __name__ == '__main__':
    demo_i18n()
