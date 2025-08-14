import asyncpg
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from .base import DatabaseManager

class PostgresManager(DatabaseManager):
    """PostgreSQL implementation of DatabaseManager"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def connect(self) -> bool:
        """Establish PostgreSQL connection pool"""
        try:
            self.pool = await asyncpg.create_pool(self.connection_string)
            await self.create_tables()
            return True
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close PostgreSQL connection pool"""
        try:
            if self.pool:
                await self.pool.close()
                self.pool = None
            return True
        except Exception as e:
            print(f"PostgreSQL disconnection failed: {e}")
            return False
    
    async def create_tables(self) -> bool:
        """Create necessary database tables"""
        try:
            async with self.pool.acquire() as conn:
                # Patients table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS patients (
                        id UUID PRIMARY KEY,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        phone VARCHAR(50),
                        date_of_birth DATE,
                        gender VARCHAR(20),
                        address TEXT,
                        emergency_contact VARCHAR(255),
                        blood_type VARCHAR(10),
                        allergies JSONB,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                """)
                
                # Medical records table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS medical_records (
                        id UUID PRIMARY KEY,
                        patient_id UUID NOT NULL,
                        record_type VARCHAR(100) NOT NULL,
                        modality VARCHAR(100),
                        diagnosis TEXT,
                        symptoms JSONB,
                        findings TEXT,
                        recommendations JSONB,
                        suggested_tests JSONB,
                        image_path TEXT,
                        confidence_score DECIMAL(5,4),
                        doctor_notes TEXT,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                    )
                """)
                
                # Appointments table (enhanced)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS appointments (
                        id UUID PRIMARY KEY,
                        patient_id UUID NOT NULL,
                        doctor_id VARCHAR(255) NOT NULL,
                        doctor_name VARCHAR(255) NOT NULL,
                        doctor_email VARCHAR(255) NOT NULL,
                        appointment_date DATE NOT NULL,
                        appointment_time TIME NOT NULL,
                        symptoms TEXT,
                        status VARCHAR(50) NOT NULL,
                        notes TEXT,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE
                    )
                """)
                
                # Create indexes for better performance
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_patients_email ON patients(email)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_medical_records_patient ON medical_records(patient_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_medical_records_type ON medical_records(record_type)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_appointments_patient ON appointments(patient_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_appointments_date ON appointments(appointment_date)")
                
            return True
            
        except Exception as e:
            print(f"Table creation failed: {e}")
            return False
    
    async def create_patient(self, patient_data: Dict[str, Any]) -> str:
        """Create a new patient record"""
        try:
            patient_id = str(uuid.uuid4())
            now = datetime.now()
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO patients (
                        id, email, name, phone, date_of_birth, gender, 
                        address, emergency_contact, blood_type, allergies, 
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, (
                    patient_id,
                    patient_data.get('email'),
                    patient_data.get('name'),
                    patient_data.get('phone'),
                    patient_data.get('date_of_birth'),
                    patient_data.get('gender'),
                    patient_data.get('address'),
                    patient_data.get('emergency_contact'),
                    patient_data.get('blood_type'),
                    json.dumps(patient_data.get('allergies', [])),
                    now, now
                ))
                
            return patient_id
            
        except Exception as e:
            print(f"Patient creation failed: {e}")
            raise
    
    async def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient by ID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM patients WHERE id = $1", patient_id)
                
                if row:
                    patient = dict(row)
                    patient['allergies'] = patient['allergies'] if patient['allergies'] else []
                    return patient
                return None
                
        except Exception as e:
            print(f"Patient retrieval failed: {e}")
            return None
    
    async def get_patient_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient by email"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM patients WHERE email = $1", email)
                
                if row:
                    patient = dict(row)
                    patient['allergies'] = patient['allergies'] if patient['allergies'] else []
                    return patient
                return None
                
        except Exception as e:
            print(f"Patient retrieval by email failed: {e}")
            return None
    
    async def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> bool:
        """Update patient information"""
        try:
            now = datetime.now()
            
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE patients SET
                        name = $1, phone = $2, date_of_birth = $3, gender = $4,
                        address = $5, emergency_contact = $6, blood_type = $7, 
                        allergies = $8, updated_at = $9
                    WHERE id = $10
                """, (
                    patient_data.get('name'),
                    patient_data.get('phone'),
                    patient_data.get('date_of_birth'),
                    patient_data.get('gender'),
                    patient_data.get('address'),
                    patient_data.get('emergency_contact'),
                    patient_data.get('blood_type'),
                    json.dumps(patient_data.get('allergies', [])),
                    now, patient_id
                ))
                
            return result != "UPDATE 0"
            
        except Exception as e:
            print(f"Patient update failed: {e}")
            return False
    
    async def delete_patient(self, patient_id: str) -> bool:
        """Delete patient record"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("DELETE FROM patients WHERE id = $1", patient_id)
            return result != "DELETE 0"
            
        except Exception as e:
            print(f"Patient deletion failed: {e}")
            return False
    
    async def add_medical_record(self, patient_id: str, record_data: Dict[str, Any]) -> str:
        """Add a new medical record"""
        try:
            record_id = str(uuid.uuid4())
            now = datetime.now()
            
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO medical_records (
                        id, patient_id, record_type, modality, diagnosis,
                        symptoms, findings, recommendations, suggested_tests,
                        image_path, confidence_score, doctor_notes, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, (
                    record_id, patient_id,
                    record_data.get('record_type'),
                    record_data.get('modality'),
                    record_data.get('diagnosis'),
                    json.dumps(record_data.get('symptoms', [])),
                    record_data.get('findings'),
                    json.dumps(record_data.get('recommendations', [])),
                    json.dumps(record_data.get('suggested_tests', [])),
                    record_data.get('image_path'),
                    record_data.get('confidence_score'),
                    record_data.get('doctor_notes'),
                    now, now
                ))
                
            return record_id
            
        except Exception as e:
            print(f"Medical record creation failed: {e}")
            raise
    
    async def get_medical_history(self, patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve patient's medical history"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM medical_records 
                    WHERE patient_id = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2
                """, patient_id, limit)
                
                records = []
                for row in rows:
                    record = dict(row)
                    record['symptoms'] = record['symptoms'] if record['symptoms'] else []
                    record['recommendations'] = record['recommendations'] if record['recommendations'] else []
                    record['suggested_tests'] = record['suggested_tests'] if record['suggested_tests'] else []
                    records.append(record)
                
                return records
                
        except Exception as e:
            print(f"Medical history retrieval failed: {e}")
            return []
    
    async def get_medical_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific medical record"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM medical_records WHERE id = $1", record_id)
                
                if row:
                    record = dict(row)
                    record['symptoms'] = record['symptoms'] if record['symptoms'] else []
                    record['recommendations'] = record['recommendations'] if record['recommendations'] else []
                    record['suggested_tests'] = record['suggested_tests'] if record['suggested_tests'] else []
                    return record
                return None
                
        except Exception as e:
            print(f"Medical record retrieval failed: {e}")
            return None
    
    async def update_medical_record(self, record_id: str, record_data: Dict[str, Any]) -> bool:
        """Update medical record"""
        try:
            now = datetime.now()
            
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE medical_records SET
                        diagnosis = $1, symptoms = $2, findings = $3,
                        recommendations = $4, suggested_tests = $5, 
                        doctor_notes = $6, updated_at = $7
                    WHERE id = $8
                """, (
                    record_data.get('diagnosis'),
                    json.dumps(record_data.get('symptoms', [])),
                    record_data.get('findings'),
                    json.dumps(record_data.get('recommendations', [])),
                    json.dumps(record_data.get('suggested_tests', [])),
                    record_data.get('doctor_notes'),
                    now, record_id
                ))
                
            return result != "UPDATE 0"
            
        except Exception as e:
            print(f"Medical record update failed: {e}")
            return False
    
    async def delete_medical_record(self, record_id: str) -> bool:
        """Delete medical record"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("DELETE FROM medical_records WHERE id = $1", record_id)
            return result != "DELETE 0"
            
        except Exception as e:
            print(f"Medical record deletion failed: {e}")
            return False
    
    async def search_patients(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search patients by name, email, or phone"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM patients 
                    WHERE name ILIKE $1 OR email ILIKE $2 OR phone ILIKE $3
                    LIMIT $4
                """, f"%{query}%", f"%{query}%", f"%{query}%", limit)
                
                patients = []
                for row in rows:
                    patient = dict(row)
                    patient['allergies'] = patient['allergies'] if patient['allergies'] else []
                    patients.append(patient)
                
                return patients
                
        except Exception as e:
            print(f"Patient search failed: {e}")
            return []
    
    async def get_patient_statistics(self, patient_id: str) -> Dict[str, Any]:
        """Get patient health statistics and trends"""
        try:
            async with self.pool.acquire() as conn:
                # Get total records
                total_records = await conn.fetchval(
                    "SELECT COUNT(*) FROM medical_records WHERE patient_id = $1", 
                    patient_id
                )
                
                # Get records by type
                records_by_type_rows = await conn.fetch("""
                    SELECT record_type, COUNT(*) 
                    FROM medical_records 
                    WHERE patient_id = $1 
                    GROUP BY record_type
                """, patient_id)
                records_by_type = {row[0]: row[1] for row in records_by_type_rows}
                
                # Get recent activity
                recent_records = await conn.fetchval("""
                    SELECT COUNT(*) FROM medical_records 
                    WHERE patient_id = $1 AND created_at >= NOW() - INTERVAL '30 days'
                """, patient_id)
                
                return {
                    "total_records": total_records,
                    "records_by_type": records_by_type,
                    "recent_records": recent_records,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Statistics retrieval failed: {e}")
            return {}
    
    async def get_condition_history(self, patient_id: str, condition: str) -> List[Dict[str, Any]]:
        """Get history of specific condition"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM medical_records 
                    WHERE patient_id = $1 AND diagnosis ILIKE $2
                    ORDER BY created_at DESC
                """, patient_id, f"%{condition}%")
                
                records = []
                for row in rows:
                    record = dict(row)
                    record['symptoms'] = record['symptoms'] if record['symptoms'] else []
                    record['recommendations'] = record['recommendations'] if record['recommendations'] else []
                    record['suggested_tests'] = record['suggested_tests'] if record['suggested_tests'] else []
                    records.append(record)
                
                return records
                
        except Exception as e:
            print(f"Condition history retrieval failed: {e}")
            return [] 