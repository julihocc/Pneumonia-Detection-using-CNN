"""
Unit tests for utility functions
"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from pneumonia_detector.utils import setup_logging, create_directories, format_bytes


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_setup_logging_default(self):
        """Test logging setup with default parameters"""
        logger = setup_logging()
        assert logger is not None
        assert logger.name == "pneumonia_detector.utils"
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            log_file = f.name
            
        try:
            logger = setup_logging(log_level="DEBUG", log_file=log_file)
            logger.info("Test message")
            
            # Check that log file was created and has content
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
                
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_create_directories(self):
        """Test directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dirs = [
                os.path.join(temp_dir, "dir1"),
                os.path.join(temp_dir, "dir2", "subdir"),
                os.path.join(temp_dir, "dir3", "sub1", "sub2")
            ]
            
            create_directories(test_dirs)
            
            for dir_path in test_dirs:
                assert os.path.exists(dir_path)
                assert os.path.isdir(dir_path)
    
    def test_format_bytes(self):
        """Test byte formatting function"""
        assert format_bytes(512) == "512.00 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024 * 1024) == "1.00 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.00 GB"
    
    @patch('pneumonia_detector.utils.tf')
    @patch('pneumonia_detector.utils.random')
    @patch('pneumonia_detector.utils.np')
    def test_set_seed(self, mock_np, mock_random, mock_tf):
        """Test seed setting function"""
        from pneumonia_detector.utils import set_seed
        
        set_seed(42)
        
        mock_random.seed.assert_called_once_with(42)
        mock_np.random.seed.assert_called_once_with(42)
        mock_tf.random.set_seed.assert_called_once_with(42)
        
        # Check environment variable
        assert os.environ.get('PYTHONHASHSEED') == '42'
    
    @patch('pneumonia_detector.utils.tf')
    def test_setup_gpu_with_gpus(self, mock_tf):
        """Test GPU setup when GPUs are available"""
        from pneumonia_detector.utils import setup_gpu
        
        # Mock GPU devices
        mock_gpu1 = MagicMock()
        mock_gpu1.name = "GPU:0"
        mock_gpu2 = MagicMock()
        mock_gpu2.name = "GPU:1"
        
        mock_tf.config.list_physical_devices.return_value = [mock_gpu1, mock_gpu2]
        
        setup_gpu()
        
        # Verify that GPU configuration functions were called
        mock_tf.config.list_physical_devices.assert_called_with('GPU')
        assert mock_tf.config.experimental.set_memory_growth.call_count == 2
        mock_tf.config.set_visible_devices.assert_called_once()
    
    @patch('pneumonia_detector.utils.tf')
    def test_setup_gpu_no_gpus(self, mock_tf):
        """Test GPU setup when no GPUs are available"""
        from pneumonia_detector.utils import setup_gpu
        
        mock_tf.config.list_physical_devices.return_value = []
        
        setup_gpu()
        
        mock_tf.config.list_physical_devices.assert_called_with('GPU')
        mock_tf.config.experimental.set_memory_growth.assert_not_called()
    
    @patch('pneumonia_detector.utils.psutil')
    @patch('pneumonia_detector.utils.platform')
    @patch('pneumonia_detector.utils.tf')
    def test_get_system_info(self, mock_tf, mock_platform, mock_psutil):
        """Test system information gathering"""
        from pneumonia_detector.utils import get_system_info
        
        # Mock system information
        mock_platform.platform.return_value = "Linux-5.4.0"
        mock_platform.python_version.return_value = "3.9.0"
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_tf.__version__ = "2.13.0"
        mock_tf.config.list_physical_devices.return_value = []
        
        info = get_system_info()
        
        assert info['platform'] == "Linux-5.4.0"
        assert info['python_version'] == "3.9.0"
        assert info['cpu_count'] == 8
        assert info['tensorflow_version'] == "2.13.0"
        assert info['gpus'] == []