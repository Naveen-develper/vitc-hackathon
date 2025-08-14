from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional
from datetime import date, datetime
from enum import Enum

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class BloodType(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

class RecordType(str, Enum):
    XRAY = "xray"
    CT_2D = "ct_2d"
    CT_3D = "ct_3d"
    MRI_2D = "mri_2d"
    MRI_3D = "mri_3d"
    ULTRASOUND = "ultrasound"
    LAB_RESULT = "lab_result"
    CONSULTATION = "consultation"
    PRESCRIPTION = "prescription"
    SURGERY = "surgery"

class Modality(str, Enum):
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    LAB = "lab"
    CLINICAL = "clinical"

# Patient Models
class PatientBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Patient's full name")
    email: EmailStr = Field(..., description="Patient's email address")
    phone: Optional[str] = Field(None, max_length=20, description="Patient's phone number")
    date_of_birth: Optional[date] = Field(None, description="Patient's date of birth")
    gender: Optional[Gender] = Field(None, description="Patient's gender")
    address: Optional[str] = Field(None, max_length=500, description="Patient's address")
    emergency_contact: Optional[str] = Field(None, max_length=100, description="Emergency contact person")
    blood_type: Optional[BloodType] = Field(None, description="Patient's blood type")
    allergies: Optional[List[str]] = Field(default=[], description="List of known allergies")

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[date] = None
    gender: Optional[Gender] = None
    address: Optional[str] = Field(None, max_length=500)
    emergency_contact: Optional[str] = Field(None, max_length=100)
    blood_type: Optional[BloodType] = None
    allergies: Optional[List[str]] = None

class PatientResponse(PatientBase):
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Medical Record Models
class MedicalRecordBase(BaseModel):
    record_type: RecordType = Field(..., description="Type of medical record")
    modality: Modality = Field(..., description="Imaging modality or record category")
    diagnosis: Optional[str] = Field(None, max_length=500, description="Medical diagnosis")
    symptoms: Optional[List[str]] = Field(default=[], description="List of symptoms")
    findings: Optional[str] = Field(None, description="Clinical findings")
    recommendations: Optional[List[str]] = Field(default=[], description="Medical recommendations")
    suggested_tests: Optional[List[str]] = Field(default=[], description="Suggested follow-up tests")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI confidence score")
    doctor_notes: Optional[str] = Field(None, description="Additional doctor notes")

class MedicalRecordCreate(MedicalRecordBase):
    # patient_id is provided in the URL path, not in request body
    pass

class MedicalRecordUpdate(BaseModel):
    diagnosis: Optional[str] = Field(None, max_length=500)
    symptoms: Optional[List[str]] = None
    findings: Optional[str] = None
    recommendations: Optional[List[str]] = None
    suggested_tests: Optional[List[str]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    doctor_notes: Optional[str] = None

class MedicalRecordResponse(MedicalRecordBase):
    id: str
    patient_id: str
    image_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Search and Filter Models
class PatientSearch(BaseModel):
    query: str = Field(..., min_length=2, max_length=100, description="Search query")
    limit: Optional[int] = Field(20, ge=1, le=100, description="Maximum number of results")

class MedicalHistoryFilter(BaseModel):
    patient_id: str = Field(..., description="Patient ID")
    record_type: Optional[RecordType] = Field(None, description="Filter by record type")
    modality: Optional[Modality] = Field(None, description="Filter by modality")
    start_date: Optional[date] = Field(None, description="Start date for timeline")
    end_date: Optional[date] = Field(None, description="End date for timeline")
    limit: Optional[int] = Field(50, ge=1, le=200, description="Maximum number of records")

class ConditionHistoryFilter(BaseModel):
    patient_id: str = Field(..., description="Patient ID")
    condition: str = Field(..., min_length=2, description="Condition to search for")

# Statistics and Summary Models
class PatientStatistics(BaseModel):
    total_records: int
    records_by_type: dict
    recent_records: int
    last_updated: datetime
    
    class Config:
        from_attributes = True

class MedicalRecordsSummary(BaseModel):
    total_records: int
    records_by_modality: dict
    recent_records: List[MedicalRecordResponse]
    common_conditions: dict
    summary_generated_at: datetime
    
    class Config:
        from_attributes = True

class PatientSummary(BaseModel):
    patient: PatientResponse
    recent_records: List[MedicalRecordResponse]
    statistics: PatientStatistics
    summary_generated_at: datetime
    
    class Config:
        from_attributes = True

# Validation Models
class PatientValidation(BaseModel):
    email: EmailStr = Field(..., description="Email to validate")
    
    @validator('email')
    def validate_email_format(cls, v):
        if not v or '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()

# Response Models for API endpoints
class PatientListResponse(BaseModel):
    patients: List[PatientResponse]
    total: int
    page: int
    limit: int

class MedicalRecordListResponse(BaseModel):
    records: List[MedicalRecordResponse]
    total: int
    patient_id: str

class SuccessResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: Optional[str] = None 