import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # Gemini AI Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Database Configuration
    DATABASE_TYPE: str = os.getenv("DATABASE_TYPE", "sqlite").lower()
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "maruthuvam_ai.db")
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads/medical_images")
    
    # Security Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not set")
            return False
        
        if cls.DATABASE_TYPE == "postgres" and not cls.DATABASE_URL:
            print("Warning: DATABASE_URL not set for PostgreSQL")
            return False
        
        return True
    
    @classmethod
    def get_database_config(cls) -> dict:
        """Get database configuration"""
        if cls.DATABASE_TYPE == "postgres":
            return {
                "type": "postgres",
                "url": cls.DATABASE_URL
            }
        else:
            return {
                "type": "sqlite",
                "path": cls.SQLITE_DB_PATH
            }
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)"""
        print("=== Maruthuvam AI Configuration ===")
        print(f"Database Type: {cls.DATABASE_TYPE}")
        print(f"Server: {cls.HOST}:{cls.PORT}")
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Upload Directory: {cls.UPLOAD_DIR}")
        print(f"Max File Size: {cls.MAX_FILE_SIZE} bytes")
        print("==================================") 