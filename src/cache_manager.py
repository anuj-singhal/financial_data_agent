"""
Cache management module for query results.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a single cache entry."""
    columns: List[str]
    rows: List[List[Any]]
    task: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - self.timestamp > timedelta(seconds=ttl_seconds)

class CacheManager:
    """
    Manages query result caching with TTL and size limits.
    """
    
    def __init__(self, max_entries: int = 100, ttl_seconds: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            max_entries: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._cache: Dict[str, CacheEntry] = {}
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        logger.info(f"CacheManager initialized with max_entries={max_entries}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Retrieve a cache entry if it exists and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            CacheEntry if found and valid, None otherwise
        """
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired(self.ttl_seconds):
                logger.debug(f"Cache hit for key: {key}")
                return entry
            else:
                logger.debug(f"Cache expired for key: {key}")
                del self._cache[key]
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    def set(self, key: str, columns: List[str], rows: List[List[Any]], task: str) -> None:
        """
        Store a new cache entry.
        
        Args:
            key: Cache key
            columns: Column names
            rows: Data rows
            task: Original task string
        """
        # Implement LRU eviction if cache is full
        if len(self._cache) >= self.max_entries:
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry: {oldest_key}")
        
        self._cache[key] = CacheEntry(columns, rows, task)
        logger.debug(f"Cache set for key: {key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all cache entries.
        
        Returns:
            Dictionary of cache summaries
        """
        summaries = {}
        for key, entry in self._cache.items():
            if not entry.is_expired(self.ttl_seconds):
                summaries[key] = {
                    "columns": entry.columns,
                    "sample_rows": entry.rows[:3],
                    "total_rows": len(entry.rows),
                    "task": entry.task,
                    "age_seconds": (datetime.now() - entry.timestamp).total_seconds()
                }
        return summaries