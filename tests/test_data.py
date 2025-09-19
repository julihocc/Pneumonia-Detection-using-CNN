"""
Unit tests for data pipeline
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

from pneumonia_detector.config import DataConfig
from pneumonia_detector.data import DataValidator


class TestDataValidator:
    """Test DataValidator class"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory structure"""
        temp_dir = tempfile.mkdtemp()
        data_root = Path(temp_dir)
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = data_root / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy images
                for i in range(5):
                    img = Image.fromarray(np.random.randint(0, 255, (150, 150), dtype=np.uint8))
                    img.save(class_dir / f"image_{i}.jpg")
        
        yield data_root
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_validate_data_structure_valid(self, temp_data_dir):
        """Test validation with valid data structure"""
        config = DataConfig()
        validator = DataValidator(config)
        
        assert validator.validate_data_structure(str(temp_data_dir)) == True
    
    def test_validate_data_structure_missing_dir(self):
        """Test validation with missing directory"""
        config = DataConfig()
        validator = DataValidator(config)
        
        assert validator.validate_data_structure("/nonexistent/path") == False
    
    def test_get_dataset_statistics(self, temp_data_dir):
        """Test dataset statistics calculation"""
        config = DataConfig()
        validator = DataValidator(config)
        
        stats = validator.get_dataset_statistics(str(temp_data_dir))
        
        # Check structure
        assert 'train' in stats
        assert 'val' in stats
        assert 'test' in stats
        
        # Check that each split has the right structure
        for split in ['train', 'val', 'test']:
            assert 'total' in stats[split]
            assert 'classes' in stats[split]
            assert 'NORMAL' in stats[split]['classes']
            assert 'PNEUMONIA' in stats[split]['classes']
            
            # Each class should have 5 images
            assert stats[split]['classes']['NORMAL'] == 5
            assert stats[split]['classes']['PNEUMONIA'] == 5
            assert stats[split]['total'] == 10


class TestDataPipeline:
    """Test DataPipeline class - simplified tests without TensorFlow"""
    
    def test_init(self):
        """Test DataPipeline initialization"""
        config = DataConfig()
        
        # This will fail without TensorFlow, but we can test the import
        try:
            from pneumonia_detector.data import DataPipeline
            pipeline = DataPipeline(config)
            assert pipeline.config == config
            assert pipeline.class_names == ["NORMAL", "PNEUMONIA"]
            assert pipeline.num_classes == 2
        except ImportError:
            pytest.skip("TensorFlow not available for testing")