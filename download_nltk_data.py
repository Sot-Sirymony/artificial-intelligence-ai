import ssl
import nltk

# SSL workaround for macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Try to download punkt_tab if available (not always needed)
try:
    nltk.download('punkt_tab')
except Exception:
    pass

print("✅ All NLTK data downloaded successfully!") 