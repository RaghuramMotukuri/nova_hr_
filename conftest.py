import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to sys.path so pytest seamlessly resolves the `src` package
sys.path.insert(0, str(Path(__file__).parent.resolve()))
