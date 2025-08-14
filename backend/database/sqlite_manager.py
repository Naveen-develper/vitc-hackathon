import sqlite3
import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from .base import DatabaseManager

class SQLiteManager(DatabaseManager):
    """SQLite implementation of DatabaseManager"""
    
    def __init__(self, db_path: str = "maruthuvam_ai.db"):
        self.db_path = db_path
        self.connection = None
    
    async def connect(self) -> bool:
        """Establish SQLite connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            await self.create_tables()
            return True
        except Exception as e:
            print(f"SQLite connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close SQLite connection"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            return True
        except Exception as e:
            print(f"SQLite disconnection failed: {e}")
            return False
    
    async def create_tables(self) -> bool:
        """Create necessary database tables"""
        try:
            cursor = self.connection.cursor()
            
            # Patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    phone TEXT,
                    date_of_birth TEXT,
                    gender TEXT,
                    address TEXT,
                    emergency_contact TEXT,
                    blood_type TEXT,
                    allergies TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Medical records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS medical_records (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    record_type TEXT NOT NULL,
                    modality TEXT,
                    diagnosis TEXT,
                    symptoms TEXT,
                    findings TEXT,
                    recommendations TEXT,
                    suggested_tests TEXT,
                    image_path TEXT,
                    confidence_score REAL,
                    doctor_notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            """)
            
            # Appointments table (enhanced)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    doctor_id TEXT NOT NULL,
                    doctor_name TEXT NOT NULL,
                    doctor_email TEXT NOT NULL,
                    appointment_date TEXT NOT NULL,
                    appointment_time TEXT NOT NULL,
                    symptoms TEXT,
                    status TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patients_email ON patients(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_medical_records_patient ON medical_records(patient_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_medical_records_type ON medical_records(record_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_appointments_patient ON appointments(patient_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_appointments_date ON appointments(appointment_date)")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"Table creation failed: {e}")
            return False
    
    async def create_patient(self, patient_data: Dict[str, Any]) -> str:
        """Create a new patient record"""
        try:
            patient_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO patients (
                    id, email, name, phone, date_of_birth, gender, 
                    address, emergency_contact, blood_type, allergies, 
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            
            self.connection.commit()
            return patient_id
            
        except Exception as e:
            print(f"Patient creation failed: {e}")
            raise
    
    async def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient by ID"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
            row = cursor.fetchone()
            
            if row:
                patient = dict(row)
                patient['allergies'] = json.loads(patient['allergies']) if patient['allergies'] else []
                return patient
            return None
            
        except Exception as e:
            print(f"Patient retrieval failed: {e}")
            return None
    
    async def get_patient_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve patient by email"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM patients WHERE email = ?", (email,))
            row = cursor.fetchone()
            
            if row:
                patient = dict(row)
                patient['allergies'] = json.loads(patient['allergies']) if patient['allergies'] else []
                return patient
            return None
            
        except Exception as e:
            print(f"Patient retrieval by email failed: {e}")
            return None
    
    async def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> bool:
        """Update patient information"""
        try:
            now = datetime.now().isoformat()
            
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE patients SET
                    name = ?, phone = ?, date_of_birth = ?, gender = ?,
                    address = ?, emergency_contact = ?, blood_type = ?, 
                    allergies = ?, updated_at = ?
                WHERE id = ?
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
            
            self.connection.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"Patient update failed: {e}")
            return False
    
    async def delete_patient(self, patient_id: str) -> bool:
        """Delete patient record"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
            self.connection.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"Patient deletion failed: {e}")
            return False
    
    async def add_medical_record(self, patient_id: str, record_data: Dict[str, Any]) -> str:
        """Add a new medical record"""
        try:
            record_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO medical_records (
                    id, patient_id, record_type, modality, diagnosis,
                    symptoms, findings, recommendations, suggested_tests,
                    image_path, confidence_score, doctor_notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            
            self.connection.commit()
            return record_id
            
        except Exception as e:
            print(f"Medical record creation failed: {e}")
            raise
    
    async def get_medical_history(self, patient_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve patient's medical history"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM medical_records 
                WHERE patient_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (patient_id, limit))
            
            records = []
            for row in cursor.fetchall():
                record = dict(row)
                record['symptoms'] = json.loads(record['symptoms']) if record['symptoms'] else []
                record['recommendations'] = json.loads(record['recommendations']) if record['recommendations'] else []
                record['suggested_tests'] = json.loads(record['suggested_tests']) if record['suggested_tests'] else []
                records.append(record)
            
            return records
            
        except Exception as e:
            print(f"Medical history retrieval failed: {e}")
            return []
    
    async def get_medical_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve specific medical record"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM medical_records WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            
            if row:
                record = dict(row)
                record['symptoms'] = json.loads(record['symptoms']) if record['symptoms'] else []
                record['recommendations'] = json.loads(record['recommendations']) if record['recommendations'] else []
                record['suggested_tests'] = json.loads(record['suggested_tests']) if record['suggested_tests'] else []
                return record
            return None
            
        except Exception as e:
            print(f"Medical record retrieval failed: {e}")
            return None
    
    async def update_medical_record(self, record_id: str, record_data: Dict[str, Any]) -> bool:
        """Update medical record"""
        try:
            now = datetime.now().isoformat()
            
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE medical_records SET
                    diagnosis = ?, symptoms = ?, findings = ?,
                    recommendations = ?, suggested_tests = ?, 
                    doctor_notes = ?, updated_at = ?
                WHERE id = ?
            """, (
                record_data.get('diagnosis'),
                json.dumps(record_data.get('symptoms', [])),
                record_data.get('findings'),
                json.dumps(record_data.get('recommendations', [])),
                json.dumps(record_data.get('suggested_tests', [])),
                record_data.get('doctor_notes'),
                now, record_id
            ))
            
            self.connection.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"Medical record update failed: {e}")
            return False
    
    async def delete_medical_record(self, record_id: str) -> bool:
        """Delete medical record"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM medical_records WHERE id = ?", (record_id,))
            self.connection.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"Medical record deletion failed: {e}")
            return False
    
    async def search_patients(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search patients by name, email, or phone"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM patients 
                WHERE name LIKE ? OR email LIKE ? OR phone LIKE ?
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
            
            patients = []
            for row in cursor.fetchall():
                patient = dict(row)
                patient['allergies'] = json.loads(patient['allergies']) if patient['allergies'] else []
                patients.append(patient)
            
            return patients
            
        except Exception as e:
            print(f"Patient search failed: {e}")
            return []
    
    async def get_patient_statistics(self, patient_id: str) -> Dict[str, Any]:
        """Get patient health statistics and trends"""
        try:
            cursor = self.connection.cursor()
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM medical_records WHERE patient_id = ?", (patient_id,))
            total_records = cursor.fetchone()[0]
            
            # Get records by type
            cursor.execute("""
                SELECT record_type, COUNT(*) 
                FROM medical_records 
                WHERE patient_id = ? 
                GROUP BY record_type
            """, (patient_id,))
            records_by_type = dict(cursor.fetchall())
            
            # Get recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM medical_records 
                WHERE patient_id = ? AND created_at >= date('now', '-30 days')
            """, (patient_id,))
            recent_records = cursor.fetchone()[0]
            
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
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM medical_records 
                WHERE patient_id = ? AND diagnosis LIKE ?
                ORDER BY created_at DESC
            """, (patient_id, f"%{condition}%"))
            
            records = []
            for row in cursor.fetchall():
                record = dict(row)
                record['symptoms'] = json.loads(record['symptoms']) if record['symptoms'] else []
                record['recommendations'] = json.loads(record['recommendations']) if record['recommendations'] else []
                record['suggested_tests'] = json.loads(record['suggested_tests']) if record['suggested_tests'] else []
                records.append(record)
            
            return records
            
        except Exception as e:
            print(f"Condition history retrieval failed: {e}")
            return [] 