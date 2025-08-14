import os
from typing import Optional
from .sqlite_manager import SQLiteManager
from .postgres_manager import PostgresManager

class DatabaseConfig:
    """Database configuration and factory class"""
    
    @staticmethod
    def get_database_manager() -> Optional[SQLiteManager | PostgresManager]:
        """
        Factory method to get the appropriate database manager
        based on environment configuration
        """
        db_type = os.getenv("DATABASE_TYPE", "sqlite").lower()
        
        if db_type == "postgres":
            connection_string = os.getenv("DATABASE_URL")
            if not connection_string:
                print("Warning: DATABASE_URL not set, falling back to SQLite")
                return SQLiteManager()
            return PostgresManager(connection_string)
        
        elif db_type == "sqlite":
            db_path = os.getenv("SQLITE_DB_PATH", "maruthuvam_ai.db")
            return SQLiteManager(db_path)
        
        else:
            print(f"Unknown database type: {db_type}, falling back to SQLite")
            return SQLiteManager()
    
    @staticmethod
    def get_connection_string() -> str:
        """Get the current database connection string for logging/debugging"""
        db_type = os.getenv("DATABASE_TYPE", "sqlite").lower()
        
        if db_type == "postgres":
            return os.getenv("DATABASE_URL", "Not set")
        else:
            db_path = os.getenv("SQLITE_DB_PATH", "maruthuvam_ai.db")
            return f"sqlite://{db_path}" 