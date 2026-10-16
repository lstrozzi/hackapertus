"""Application configuration module."""

import os

class AppConfig:
    """Centralized application configuration."""
    # attributes
    _testing: bool = None

    def __init__(self):
        self._testing = None
        
    @property
    def testing(self) -> bool:
        """Get testing mode status."""
        if self._testing is None:
            # Check environment variable first, then Flask config if available
            self._testing = os.getenv('FLASK_TESTING', 'False').lower() == 'true'
        return self._testing
    
    def set_testing(self, value: bool):
        """Set testing mode (called by Flask app initialization)."""
        self._testing = value

# Global instance
app_config = AppConfig()
