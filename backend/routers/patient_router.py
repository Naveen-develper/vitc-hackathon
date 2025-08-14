from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Optional
import os
from services.patient_service import PatientService
from services.medical_records_service import MedicalRecordsService
from models.patient_models import (
    PatientCreate, PatientUpdate, PatientResponse, PatientSearch,
    MedicalRecordCreate, MedicalRecordUpdate, MedicalRecordResponse,
    MedicalHistoryFilter, ConditionHistoryFilter, PatientListResponse,
    MedicalRecordListResponse, SuccessResponse, ErrorResponse
)

router = APIRouter(prefix="/api/patients", tags=["Patients"])

# Dependency to get services
async def get_patient_service():
    service = PatientService()
    await service.initialize()
    try:
        yield service
    finally:
        await service.cleanup()

async def get_medical_records_service():
    service = MedicalRecordsService()
    await service.initialize()
    try:
        yield service
    finally:
        await service.cleanup()

# Patient Management Endpoints
@router.post("/", response_model=PatientResponse)
async def create_patient(
    patient: PatientCreate,
    service: PatientService = Depends(get_patient_service)
):
    """Create a new patient"""
    try:
        return await service.create_patient(patient.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: str,
    service: PatientService = Depends(get_patient_service)
):
    """Get patient by ID"""
    patient = await service.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.get("/email/{email}", response_model=PatientResponse)
async def get_patient_by_email(
    email: str,
    service: PatientService = Depends(get_patient_service)
):
    """Get patient by email"""
    patient = await service.get_patient_by_email(email)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: str,
    patient_update: PatientUpdate,
    service: PatientService = Depends(get_patient_service)
):
    """Update patient information"""
    try:
        # Filter out None values
        update_data = {k: v for k, v in patient_update.dict().items() if v is not None}
        return await service.update_patient(patient_id, update_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{patient_id}", response_model=SuccessResponse)
async def delete_patient(
    patient_id: str,
    service: PatientService = Depends(get_patient_service)
):
    """Delete patient and all associated records"""
    try:
        success = await service.delete_patient(patient_id)
        if success:
            return SuccessResponse(success=True, message="Patient deleted successfully")
        else:
            raise HTTPException(status_code=400, detail="Failed to delete patient")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/search", response_model=List[PatientResponse])
async def search_patients(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    service: PatientService = Depends(get_patient_service)
):
    """Search patients by name, email, or phone"""
    patients = await service.search_patients(query, limit)
    return patients

@router.get("/{patient_id}/statistics")
async def get_patient_statistics(
    patient_id: str,
    service: PatientService = Depends(get_patient_service)
):
    """Get patient health statistics"""
    stats = await service.get_patient_statistics(patient_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Patient not found")
    return stats

@router.get("/{patient_id}/summary")
async def get_patient_summary(
    patient_id: str,
    service: PatientService = Depends(get_patient_service)
):
    """Get comprehensive patient summary"""
    summary = await service.get_patient_summary(patient_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Patient not found")
    return summary

# Medical Records Endpoints
@router.post("/{patient_id}/medical-records", response_model=MedicalRecordResponse)
async def create_medical_record(
    patient_id: str,
    record_data: MedicalRecordCreate,
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Create a new medical record for a patient"""
    try:
        # Validate patient exists
        patient_service = PatientService()
        await patient_service.initialize()
        patient = await patient_service.get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        await patient_service.cleanup()
        
        # Create medical record
        record_dict = record_data.dict()
        record_dict['patient_id'] = patient_id
        
        return await service.create_medical_record(patient_id, record_dict, None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{patient_id}/medical-records", response_model=MedicalRecordListResponse)
async def get_medical_history(
    patient_id: str,
    record_type: Optional[str] = Query(None, description="Filter by record type"),
    modality: Optional[str] = Query(None, description="Filter by modality"),
    limit: int = Query(50, ge=1, le=200, description="Maximum records"),
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Get patient's medical history with optional filtering"""
    try:
        records = await service.get_medical_history(patient_id, limit, record_type)
        
        # Filter by modality if specified
        if modality:
            records = [r for r in records if r.get('modality') == modality]
        
        return MedicalRecordListResponse(
            records=records,
            total=len(records),
            patient_id=patient_id
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/medical-records/{record_id}", response_model=MedicalRecordResponse)
async def get_medical_record(
    record_id: str,
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Get specific medical record by ID"""
    record = await service.get_medical_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Medical record not found")
    return record

@router.put("/medical-records/{record_id}", response_model=MedicalRecordResponse)
async def update_medical_record(
    record_id: str,
    record_update: MedicalRecordUpdate,
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Update medical record"""
    try:
        # Filter out None values
        update_data = {k: v for k, v in record_update.dict().items() if v is not None}
        return await service.update_medical_record(record_id, update_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/medical-records/{record_id}", response_model=SuccessResponse)
async def delete_medical_record(
    record_id: str,
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Delete medical record"""
    try:
        success = await service.delete_medical_record(record_id)
        if success:
            return SuccessResponse(success=True, message="Medical record deleted successfully")
        else:
            raise HTTPException(status_code=400, detail="Failed to delete medical record")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{patient_id}/medical-records/condition/{condition}")
async def get_condition_history(
    patient_id: str,
    condition: str,
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Get history of specific condition for a patient"""
    records = await service.get_records_by_condition(patient_id, condition)
    return {
        "patient_id": patient_id,
        "condition": condition,
        "records": records,
        "total": len(records)
    }

@router.get("/{patient_id}/medical-records/modality/{modality}")
async def get_records_by_modality(
    patient_id: str,
    modality: str,
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Get all records for a specific imaging modality"""
    records = await service.get_records_by_modality(patient_id, modality)
    return {
        "patient_id": patient_id,
        "modality": modality,
        "records": records,
        "total": len(records)
    }

@router.get("/{patient_id}/medical-records/timeline")
async def get_records_timeline(
    patient_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Get medical records within a date range"""
    records = await service.get_records_timeline(patient_id, start_date, end_date)
    return {
        "patient_id": patient_id,
        "start_date": start_date,
        "end_date": end_date,
        "records": records,
        "total": len(records)
    }

@router.get("/{patient_id}/medical-records/summary")
async def get_records_summary(
    patient_id: str,
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Get summary of all medical records for a patient"""
    summary = await service.get_records_summary(patient_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Patient not found")
    return summary

# Image Access Endpoint
@router.get("/medical-records/{record_id}/image")
async def get_medical_image(
    record_id: str,
    service: MedicalRecordsService = Depends(get_medical_records_service)
):
    """Get medical image for a specific record"""
    image_path = await service.get_image_path(record_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return FileResponse(image_path, media_type="image/*")

# Health Check Endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "patient-management"} 