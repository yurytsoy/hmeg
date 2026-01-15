"""
Manager for LanguageTool server lifecycle.

This module provides a singleton manager class for handling the LanguageTool server's lifetime
to avoid brittle global variable initialization at module import time.
"""
from __future__ import annotations

import atexit
import language_tool_python as ltp
import requests

from .usecases import is_port_in_use


class LanguageToolManager:
    """
    Singleton manager for LanguageTool server lifecycle.
    
    This class ensures that only one LanguageTool instance is created and properly
    managed throughout the application lifetime.
    """
    
    _instance: LanguageToolManager | None = None
    _language_tool: ltp.LanguageTool | None = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            atexit.register(cls._instance._cleanup)
        return cls._instance
    
    def get_language_tool(self) -> ltp.LanguageTool:
        """
        Get or create `LanguageTool` instance.
        
        Returns
        -------
        ltp.LanguageTool
            The LanguageTool instance for grammar checking.
        """
        if self._language_tool is None:
            self._language_tool = self._create_language_tool()
        return self._language_tool
    
    def _create_language_tool(self) -> ltp.LanguageTool:
        """
        Create a new LanguageTool instance.
        
        Returns
        -------
        ltp.LanguageTool
            A new LanguageTool instance connected to a local server.
            
        Raises
        ------
        RuntimeError
            If all tested ports are in use by other services.
        """
        def is_lt_server_running(port: int) -> bool:
            """Check if a LanguageTool server is running on the specified port."""
            LT_URL = f"http://localhost:{port}/v2/check"
            
            try:
                args = "text=foo&language=en"
                response = requests.post(f"{LT_URL}?{args}", timeout=2)
                return response.status_code == 200
            except requests.RequestException:
                return False
        
        # Note: the code below inherently assumes that LT is trying to use ports starting from `LanguageTool._MIN_PORT`
        ports = [ltp.LanguageTool._MIN_PORT + k for k in range(10)]
        for port in ports:
            if is_port_in_use(port):
                if is_lt_server_running(port):
                    return ltp.LanguageTool('en-US', remote_server=f"localhost:{port}")
                # Port is in use but not running LanguageTool, try next port
                continue
            
            # Port is free, try to start a new LanguageTool server here.
            return ltp.LanguageTool('en-US', host='localhost')
        
        raise RuntimeError(f"All ports are in use by other services (tested ports: {ports})")
    
    def _cleanup(self) -> None:
        """Clean up the LanguageTool instance on exit."""
        if self._language_tool is not None:
            try:
                self._language_tool.close()
            except Exception:
                # Suppress any errors during cleanup
                pass
            finally:
                self._language_tool = None
    
    def close(self) -> None:
        """Explicitly close the LanguageTool instance."""
        self._cleanup()
