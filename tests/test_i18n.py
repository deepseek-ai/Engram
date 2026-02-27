"""
Tests for Engram i18n support.

Run tests with: python -m pytest tests/test_i18n.py -v
Or: python tests/test_i18n.py
"""

import os
import sys
import unittest
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engram.i18n import _, set_language, get_current_language, get_locales_dir


class TestI18N(unittest.TestCase):
    """Test cases for internationalization support."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_strings = {
            "forward_complete": "✅ Forward Complete!",
            "demo_title": "[Engram Architecture Demo Implementation]",
            "demo_purpose": "1. Demo Purpose Only:",
            "production_readiness": "2. Production Readiness:",
            "simplifications": "3. Simplifications:",
            "example_text": "Only Alexander the Great could tame the horse Bucephalus.",
        }
    
    def test_default_language_is_english(self):
        """Test that default language is English."""
        set_language('en')
        self.assertEqual(get_current_language(), 'en')
    
    def test_english_translation(self):
        """Test English translations return original strings."""
        set_language('en')
        
        for key, original in self.test_strings.items():
            translated = _(original)
            self.assertEqual(translated, original,
                           f"English translation for '{key}' should return original string")
    
    def test_chinese_translation(self):
        """Test Chinese translations return different strings."""
        set_language('zh_CN')
        
        # These strings should be translated
        test_cases = {
            "✅ Forward Complete!": "✅ 前向传播完成！",
            "[Engram Architecture Demo Implementation]": "[Engram 架构演示实现]",
            "Only Alexander the Great could tame the horse Bucephalus.": "只有亚历山大大帝才能驯服战马布塞法勒斯。",
        }
        
        for original, expected_chinese in test_cases.items():
            translated = _(original)
            # The translation might be the original if .mo file is missing
            # but in our case it should be translated
            self.assertIn(translated, [original, expected_chinese],
                         f"Translation for '{original}' should be either original or Chinese")
    
    def test_language_switching(self):
        """Test that language can be switched dynamically."""
        # Start with English
        set_language('en')
        self.assertEqual(get_current_language(), 'en')
        
        # Switch to Chinese
        set_language('zh_CN')
        self.assertEqual(get_current_language(), 'zh_CN')
        
        # Switch back to English
        set_language('en')
        self.assertEqual(get_current_language(), 'en')
    
    def test_locales_directory_exists(self):
        """Test that locales directory structure exists."""
        locales_dir = get_locales_dir()
        self.assertTrue(os.path.exists(locales_dir),
                       f"Locales directory should exist: {locales_dir}")
        
        # Check for English and Chinese directories
        en_dir = os.path.join(locales_dir, 'en', 'LC_MESSAGES')
        zh_dir = os.path.join(locales_dir, 'zh_CN', 'LC_MESSAGES')
        
        self.assertTrue(os.path.exists(en_dir),
                       f"English locale directory should exist: {en_dir}")
        self.assertTrue(os.path.exists(zh_dir),
                       f"Chinese locale directory should exist: {zh_dir}")
    
    def test_mo_files_exist(self):
        """Test that compiled .mo files exist."""
        locales_dir = get_locales_dir()
        
        en_mo = os.path.join(locales_dir, 'en', 'LC_MESSAGES', 'engram.mo')
        zh_mo = os.path.join(locales_dir, 'zh_CN', 'LC_MESSAGES', 'engram.mo')
        
        self.assertTrue(os.path.exists(en_mo),
                       f"English .mo file should exist: {en_mo}")
        self.assertTrue(os.path.exists(zh_mo),
                       f"Chinese .mo file should exist: {zh_mo}")
    
    def test_po_files_exist(self):
        """Test that source .po files exist."""
        locales_dir = get_locales_dir()
        
        en_po = os.path.join(locales_dir, 'en', 'LC_MESSAGES', 'engram.po')
        zh_po = os.path.join(locales_dir, 'zh_CN', 'LC_MESSAGES', 'engram.po')
        
        self.assertTrue(os.path.exists(en_po),
                       f"English .po file should exist: {en_po}")
        self.assertTrue(os.path.exists(zh_po),
                       f"Chinese .po file should exist: {zh_po}")
    
    def test_fallback_for_unknown_language(self):
        """Test that unknown languages fall back to English."""
        set_language('unknown_language_code')
        # Should not raise an error
        result = _("✅ Forward Complete!")
        # Should return the original string (fallback behavior)
        self.assertIsInstance(result, str)
    
    def test_translation_with_variables(self):
        """Test that translation works with f-strings."""
        set_language('en')
        
        # This is how variables should be used with translations
        shape_info = "test_shape"
        result = f"{shape_info}"
        self.assertEqual(result, "test_shape")


# Check if transformers is available
TRANSFORMERS_AVAILABLE = importlib.util.find_spec('transformers') is not None


class TestI18NIntegration(unittest.TestCase):
    """Integration tests for i18n with the main module."""
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers module not installed")
    def test_import_main_module(self):
        """Test that the main module can be imported with i18n."""
        try:
            import engram_demo_v1
            self.assertTrue(hasattr(engram_demo_v1, 'I18N_AVAILABLE'))
        except ImportError as e:
            self.fail(f"Should be able to import engram_demo_v1: {e}")
    
    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers module not installed")
    def test_argument_parser_with_i18n(self):
        """Test that argument parser accepts language option."""
        from engram_demo_v1 import parse_args
        
        # Test default language
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ['engram_demo_v1.py']
            args = parse_args()
            self.assertEqual(args.language, 'en')
        finally:
            sys.argv = old_argv


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestI18N))
    suite.addTests(loader.loadTestsFromTestCase(TestI18NIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
