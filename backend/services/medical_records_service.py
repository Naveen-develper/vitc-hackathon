from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import os
from database.config import DatabaseConfig

class MedicalRecordsService:
    """Service layer for medical records management"""
    
    def __init__(self):
        self.db = DatabaseConfig.get_database_manager()
        self.upload_dir = "uploads/medical_images"
        self._ensure_upload_dir()
    
    def _ensure_upload_dir(self):
        """Ensure upload directory exists"""
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def initialize(self):
        """Initialize database connection"""
        return await self.db.connect()
    
    async def cleanup(self):
        """Cleanup database connection"""
        return await self.db.disconnect()
    
    async def create_medical_record(self, patient_id: str, record_data: Dict[str, Any], image_file=None) -> Dict[str, Any]:
        """Create a new medical record with optional image"""
        try:
            # Validate required fields
            required_fields = ['record_type', 'modality']
            for field in required_fields:
                if not record_data.get(field):
                    raise ValueError(f"Missing required field: {field}")
            
            # Handle image upload if provided
            image_path = None
            if image_file:
                image_path = await self._save_medical_image(patient_id, image_file, record_data['modality'])
                record_data['image_path'] = image_path
            
            # Add metadata
            record_data['created_at'] = datetime.now().isoformat()
            record_data['updated_at'] = datetime.now().isoformat()
            
            # Create record in database
            record_id = await self.db.add_medical_record(patient_id, record_data)
            
            # Return created record
            return await self.db.get_medical_record(record_id)
            
        except Exception as e:
            # Cleanup image if record creation failed
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
            raise Exception(f"Failed to create medical record: {str(e)}")
    
    async def get_medical_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get medical record by ID"""
        try:
            return await self.db.get_medical_record(record_id)
        except Exception as e:
            print(f"Error retrieving medical record: {e}")
            return None
    
    async def get_medical_history(self, patient_id: str, limit: int = 50, record_type: str = None) -> List[Dict[str, Any]]:
        """Get patient's medical history with optional filtering"""
        try:
            records = await self.db.get_medical_history(patient_id, limit)
            
            # Filter by record type if specified
            if record_type:
                records = [r for r in records if r.get('record_type') == record_type]
            
            return records
            
        except Exception as e:
            print(f"Error retrieving medical history: {e}")
            return []
    
    async def update_medical_record(self, record_id: str, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update medical record"""
        try:
            # Check if record exists
            existing_record = await self.db.get_medical_record(record_id)
            if not existing_record:
                raise ValueError(f"Medical record with ID {record_id} not found")
            
            # Add update timestamp
            record_data['updated_at'] = datetime.now().isoformat()
            
            # Update record
            success = await self.db.update_medical_record(record_id, record_data)
            if not success:
                raise Exception("Failed to update medical record")
            
            # Return updated record
            return await self.db.get_medical_record(record_id)
            
        except Exception as e:
            raise Exception(f"Failed to update medical record: {str(e)}")
    
    async def delete_medical_record(self, record_id: str) -> bool:
        """Delete medical record and associated image"""
        try:
            # Get record to find image path
            record = await self.db.get_medical_record(record_id)
            if not record:
                raise ValueError(f"Medical record with ID {record_id} not found")
            
            # Delete associated image if exists
            if record.get('image_path') and os.path.exists(record['image_path']):
                os.remove(record['image_path'])
            
            # Delete record from database
            return await self.db.delete_medical_record(record_id)
            
        except Exception as e:
            raise Exception(f"Failed to delete medical record: {str(e)}")
    
    async def get_records_by_condition(self, patient_id: str, condition: str) -> List[Dict[str, Any]]:
        """Get all records for a specific condition"""
        try:
            return await self.db.get_condition_history(patient_id, condition)
        except Exception as e:
            print(f"Error getting condition history: {e}")
            return []
    
    async def get_records_by_modality(self, patient_id: str, modality: str) -> List[Dict[str, Any]]:
        """Get all records for a specific imaging modality"""
        try:
            records = await self.db.get_medical_history(patient_id, limit=100)
            return [r for r in records if r.get('modality') == modality]
        except Exception as e:
            print(f"Error getting records by modality: {e}")
            return []
    
    async def get_records_timeline(self, patient_id: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get medical records within a date range"""
        try:
            records = await self.db.get_medical_history(patient_id, limit=100)
            
            # Filter by date range if provided
            if start_date or end_date:
                filtered_records = []
                for record in records:
                    record_date = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00'))
                    
                    if start_date:
                        start = datetime.fromisoformat(start_date)
                        if record_date < start:
                            continue
                    
                    if end_date:
                        end = datetime.fromisoformat(end_date)
                        if record_date > end:
                            continue
                    
                    filtered_records.append(record)
                
                return filtered_records
            
            return records
            
        except Exception as e:
            print(f"Error getting records timeline: {e}")
            return []
    
    async def _save_medical_image(self, patient_id: str, image_file, modality: str) -> str:
        """Save medical image to disk"""
        try:
            # Create patient-specific directory
            patient_dir = os.path.join(self.upload_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_extension = os.path.splitext(image_file.filename)[1] if hasattr(image_file, 'filename') else '.jpg'
            filename = f"{modality}_{timestamp}{file_extension}"
            
            file_path = os.path.join(patient_dir, filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                if hasattr(image_file, 'file'):
                    # FastAPI UploadFile
                    buffer.write(image_file.file.read())
                else:
                    # Direct file object
                    buffer.write(image_file.read())
            
            return file_path
            
        except Exception as e:
            raise Exception(f"Failed to save medical image: {str(e)}")
    
    async def get_image_path(self, record_id: str) -> Optional[str]:
        """Get the file path for a medical image"""
        try:
            record = await self.db.get_medical_record(record_id)
            if record and record.get('image_path'):
                return record['image_path'] if os.path.exists(record['image_path']) else None
            return None
        except Exception as e:
            print(f"Error getting image path: {e}")
            return None
    
    async def get_records_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get a summary of all medical records for a patient"""
        try:
            # Get all records
            all_records = await self.db.get_medical_history(patient_id, limit=1000)
            
            # Group by modality
            records_by_modality = {}
            for record in all_records:
                modality = record.get('modality', 'Unknown')
                if modality not in records_by_modality:
                    records_by_modality[modality] = []
                records_by_modality[modality].append(record)
            
            # Get recent activity
            recent_records = all_records[:10] if all_records else []
            
            # Get conditions summary
            conditions = {}
            for record in all_records:
                diagnosis = record.get('diagnosis', 'Unknown')
                if diagnosis not in conditions:
                    conditions[diagnosis] = 0
                conditions[diagnosis] += 1
            
            return {
                "total_records": len(all_records),
                "records_by_modality": {k: len(v) for k, v in records_by_modality.items()},
                "recent_records": recent_records,
                "common_conditions": dict(sorted(conditions.items(), key=lambda x: x[1], reverse=True)[:5]),
                "summary_generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting records summary: {e}")
            return {} 