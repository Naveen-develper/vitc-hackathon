from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

class DatabaseManager(ABC):
    """Abstract base class for database operations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish database connection"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close database connection"""
        pass
    
    @abstractmethod
    async def create_tables(self) -> bool:
        """Create necessary database tables"""
        pass
    
    # Patient operations
    @abstractmethod
    async def create_patient(self, patient_data: Dict[str, Any]) -> str:
        """Create a new patient record"""
        pass
    
    @abstractmethod
    async def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient by ID"""
        pass
    
    @abstractmethod
    async def get_patient_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient by email"""
        pass
    
    @abstractmethod
    async def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> bool:
        """Update patient information"""
        pass
    
    @abstractmethod
    async def delete_patient(self, patient_id: str) -> bool:
        """Delete patient record"""
        pass
    
    # Medical history operations
    @abstractmethod
    async def add_medical_record(self, patient_id: str, record_data: Dict[str, Any]) -> str:
        """Add a new medical record"""
        pass
    
    @abstractmethod
    async def get_medical_history(self, patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve patient's medical history"""
        pass
    
    @abstractmethod
    async def get_medical_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific medical record"""
        pass
    
    @abstractmethod
    async def update_medical_record(self, record_id: str, record_data: Dict[str, Any]) -> bool:
        """Update medical record"""
        pass
    
    @abstractmethod
    async def delete_medical_record(self, record_id: str) -> bool:
        """Delete medical record"""
        pass
    
    # Search and analytics
    @abstractmethod
    async def search_patients(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search patients by name, email, or phone"""
        pass
    
    @abstractmethod
    async def get_patient_statistics(self, patient_id: str) -> Dict[str, Any]:
        """Get patient health statistics and trends"""
        pass
    
    @abstractmethod
    async def get_condition_history(self, patient_id: str, condition: str) -> List[Dict[str, Any]]:
        """Get history of specific condition"""
        pass 