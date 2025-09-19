"""
Test configuration for pytest
"""
import pytest
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def sample_config():
    """Provide a sample configuration for tests"""
    from pneumonia_detector.config import Config
    return Config()


@pytest.fixture
def temp_image():
    """Create a temporary test image"""
    import tempfile
    import numpy as np
    from PIL import Image
    
    # Create a random image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name)
        yield f.name
    
    # Cleanup
    import os
    try:
        os.unlink(f.name)
    except:
        pass