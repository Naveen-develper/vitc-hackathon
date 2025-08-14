from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from database.config import DatabaseConfig

class PatientService:
    """Service layer for patient management operations"""
    
    def __init__(self):
        self.db = DatabaseConfig.get_database_manager()
    
    async def initialize(self):
        """Initialize database connection"""
        return await self.db.connect()
    
    async def cleanup(self):
        """Cleanup database connection"""
        return await self.db.disconnect()
    
    async def create_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new patient with validation"""
        try:
            # Validate required fields
            required_fields = ['email', 'name']
            for field in required_fields:
                if not patient_data.get(field):
                    raise ValueError(f"Missing required field: {field}")
            
            # Check if patient already exists
            existing_patient = await self.db.get_patient_by_email(patient_data['email'])
            if existing_patient:
                raise ValueError(f"Patient with email {patient_data['email']} already exists")
            
            # Create patient
            patient_id = await self.db.create_patient(patient_data)
            
            # Return created patient
            return await self.db.get_patient(patient_id)
            
        except Exception as e:
            raise Exception(f"Failed to create patient: {str(e)}")
    
    async def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient by ID"""
        try:
            return await self.db.get_patient(patient_id)
        except Exception as e:
            print(f"Error retrieving patient: {e}")
            return None
    
    async def get_patient_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get patient by email"""
        try:
            return await self.db.get_patient_by_email(email)
        except Exception as e:
            print(f"Error retrieving patient by email: {e}")
            return None
    
    async def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update patient information"""
        try:
            # Check if patient exists
            existing_patient = await self.db.get_patient(patient_id)
            if not existing_patient:
                raise ValueError(f"Patient with ID {patient_id} not found")
            
            # Update patient
            success = await self.db.update_patient(patient_id, patient_data)
            if not success:
                raise Exception("Failed to update patient")
            
            # Return updated patient
            return await self.db.get_patient(patient_id)
            
        except Exception as e:
            raise Exception(f"Failed to update patient: {str(e)}")
    
    async def delete_patient(self, patient_id: str) -> bool:
        """Delete patient and all associated records"""
        try:
            # Check if patient exists
            existing_patient = await self.db.get_patient(patient_id)
            if not existing_patient:
                raise ValueError(f"Patient with ID {patient_id} not found")
            
            # Delete patient (this will cascade to medical records and appointments)
            return await self.db.delete_patient(patient_id)
            
        except Exception as e:
            raise Exception(f"Failed to delete patient: {str(e)}")
    
    async def search_patients(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search patients by name, email, or phone"""
        try:
            if not query or len(query.strip()) < 2:
                return []
            
            return await self.db.search_patients(query.strip(), limit)
            
        except Exception as e:
            print(f"Error searching patients: {e}")
            return []
    
    async def get_patient_statistics(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient health statistics"""
        try:
            # Get basic statistics
            stats = await self.db.get_patient_statistics(patient_id)
            
            # Get patient info
            patient = await self.db.get_patient(patient_id)
            if patient:
                stats['patient_info'] = {
                    'name': patient['name'],
                    'email': patient['email'],
                    'age': self._calculate_age(patient.get('date_of_birth')),
                    'blood_type': patient.get('blood_type'),
                    'allergies': patient.get('allergies', [])
                }
            
            return stats
            
        except Exception as e:
            print(f"Error getting patient statistics: {e}")
            return {}
    
    async def get_condition_history(self, patient_id: str, condition: str) -> List[Dict[str, Any]]:
        """Get history of specific condition for a patient"""
        try:
            return await self.db.get_condition_history(patient_id, condition)
        except Exception as e:
            print(f"Error getting condition history: {e}")
            return []
    
    def _calculate_age(self, date_of_birth: str) -> Optional[int]:
        """Calculate age from date of birth"""
        try:
            if not date_of_birth:
                return None
            
            dob = datetime.fromisoformat(date_of_birth.replace('Z', '+00:00'))
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except:
            return None
    
    async def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get a comprehensive patient summary"""
        try:
            patient = await self.db.get_patient(patient_id)
            if not patient:
                return {}
            
            # Get recent medical records
            recent_records = await self.db.get_medical_history(patient_id, limit=5)
            
            # Get statistics
            stats = await self.db.get_patient_statistics(patient_id)
            
            return {
                "patient": patient,
                "recent_records": recent_records,
                "statistics": stats,
                "summary_generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting patient summary: {e}")
            return {} 