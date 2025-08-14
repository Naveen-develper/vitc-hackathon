# 🏥 Patient Management System - Maruthuvam AI

## 📋 Overview

The Patient Management System is a comprehensive module that adds persistent storage and patient history tracking to the Maruthuvam AI platform. It provides a modular, scalable architecture that can work with both SQLite (development) and PostgreSQL (production) databases.

## 🏗️ Architecture

### **Modular Design Pattern**
```
backend/
├── database/                 # Database abstraction layer
│   ├── __init__.py          # Module initialization
│   ├── base.py              # Abstract base class
│   ├── sqlite_manager.py    # SQLite implementation
│   ├── postgres_manager.py  # PostgreSQL implementation
│   └── config.py            # Database configuration
├── services/                 # Business logic layer
│   ├── patient_service.py   # Patient management
│   └── medical_records_service.py  # Medical records
├── models/                   # Data validation
│   └── patient_models.py    # Pydantic models
├── routers/                  # API endpoints
│   └── patient_router.py    # Patient API routes
└── main.py                  # Main application
```

### **Key Components**

1. **Database Layer**: Abstract interface with multiple implementations
2. **Service Layer**: Business logic and data validation
3. **Model Layer**: Pydantic models for data validation
4. **Router Layer**: FastAPI endpoints for patient management
5. **Configuration**: Environment-based configuration management

## 🗄️ Database Schema

### **Patients Table**
```sql
CREATE TABLE patients (
    id TEXT PRIMARY KEY,           -- UUID
    email TEXT UNIQUE NOT NULL,    -- Patient email
    name TEXT NOT NULL,            -- Full name
    phone TEXT,                    -- Phone number
    date_of_birth TEXT,            -- Date of birth
    gender TEXT,                   -- Gender
    address TEXT,                  -- Address
    emergency_contact TEXT,        -- Emergency contact
    blood_type TEXT,               -- Blood type
    allergies TEXT,                -- JSON array of allergies
    created_at TEXT NOT NULL,      -- Creation timestamp
    updated_at TEXT NOT NULL       -- Last update timestamp
);
```

### **Medical Records Table**
```sql
CREATE TABLE medical_records (
    id TEXT PRIMARY KEY,           -- UUID
    patient_id TEXT NOT NULL,      -- Foreign key to patients
    record_type TEXT NOT NULL,     -- Type of record
    modality TEXT,                 -- Imaging modality
    diagnosis TEXT,                -- Medical diagnosis
    symptoms TEXT,                 -- JSON array of symptoms
    findings TEXT,                 -- Clinical findings
    recommendations TEXT,          -- JSON array of recommendations
    suggested_tests TEXT,          -- JSON array of suggested tests
    image_path TEXT,               -- Path to medical image
    confidence_score REAL,         -- AI confidence (0.0-1.0)
    doctor_notes TEXT,             -- Additional notes
    created_at TEXT NOT NULL,      -- Creation timestamp
    updated_at TEXT NOT NULL,      -- Last update timestamp
    FOREIGN KEY (patient_id) REFERENCES patients (id)
);
```

### **Enhanced Appointments Table**
```sql
CREATE TABLE appointments (
    id TEXT PRIMARY KEY,           -- UUID
    patient_id TEXT NOT NULL,      -- Foreign key to patients
    doctor_id TEXT NOT NULL,       -- Doctor identifier
    doctor_name TEXT NOT NULL,     -- Doctor name
    doctor_email TEXT NOT NULL,    -- Doctor email
    appointment_date TEXT NOT NULL, -- Appointment date
    appointment_time TEXT NOT NULL, -- Appointment time
    symptoms TEXT,                 -- Patient symptoms
    status TEXT NOT NULL,          -- Appointment status
    notes TEXT,                    -- Additional notes
    created_at TEXT NOT NULL,      -- Creation timestamp
    updated_at TEXT NOT NULL,      -- Last update timestamp
    FOREIGN KEY (patient_id) REFERENCES patients (id)
);
```

## 🔌 API Endpoints

### **Patient Management**

#### `POST /api/patients/`
Create a new patient
```json
{
  "name": "John Doe",
  "email": "john.doe@example.com",
  "phone": "+91-98765-43210",
  "date_of_birth": "1990-01-01",
  "gender": "male",
  "blood_type": "O+",
  "allergies": ["Penicillin", "Shellfish"]
}
```

#### `GET /api/patients/{patient_id}`
Get patient by ID

#### `GET /api/patients/email/{email}`
Get patient by email

#### `PUT /api/patients/{patient_id}`
Update patient information

#### `DELETE /api/patients/{patient_id}`
Delete patient and all associated records

#### `GET /api/patients/search?query={search_term}&limit={limit}`
Search patients by name, email, or phone

#### `GET /api/patients/{patient_id}/statistics`
Get patient health statistics

#### `GET /api/patients/{patient_id}/summary`
Get comprehensive patient summary

### **Medical Records**

#### `POST /api/patients/{patient_id}/medical-records`
Create a new medical record
```json
{
  "record_type": "xray",
  "modality": "xray",
  "diagnosis": "Pneumonia",
  "symptoms": ["Cough", "Fever", "Difficulty breathing"],
  "findings": "Bilateral infiltrates in lower lobes",
  "recommendations": ["Antibiotics", "Rest", "Follow-up in 1 week"],
  "suggested_tests": ["CBC", "Chest X-ray follow-up"],
  "confidence_score": 0.85
}
```

#### `GET /api/patients/{patient_id}/medical-records`
Get patient's medical history with optional filtering

#### `GET /api/medical-records/{record_id}`
Get specific medical record

#### `PUT /api/medical-records/{record_id}`
Update medical record

#### `DELETE /api/medical-records/{record_id}`
Delete medical record

#### `GET /api/patients/{patient_id}/medical-records/condition/{condition}`
Get history of specific condition

#### `GET /api/patients/{patient_id}/medical-records/modality/{modality}`
Get records by imaging modality

#### `GET /api/patients/{patient_id}/medical-records/timeline`
Get records within date range

#### `GET /api/patients/{patient_id}/medical-records/summary`
Get summary of all medical records

#### `GET /api/medical-records/{record_id}/image`
Get medical image for a record

## 🚀 Setup and Configuration

### **1. Environment Variables**
Create a `.env` file in the backend directory:

```bash
# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Database Configuration
DATABASE_TYPE=sqlite  # Options: sqlite, postgres
SQLITE_DB_PATH=maruthuvam_ai.db

# PostgreSQL Configuration (if using postgres)
# DATABASE_URL=postgresql://username:password@localhost:5432/maruthuvam_ai

# Server Configuration
HOST=0.0.0.0
PORT=8001
DEBUG=true

# File Upload Configuration
MAX_FILE_SIZE=10485760  # 10MB in bytes
UPLOAD_DIR=uploads/medical_images
```

### **2. Install Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

### **3. Database Setup**
The system automatically creates the necessary tables on first run.

**For SQLite (Default):**
- No additional setup required
- Database file created automatically

**For PostgreSQL:**
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb maruthuvam_ai

# Set environment variable
export DATABASE_URL="postgresql://username:password@localhost:5432/maruthuvam_ai"
```

### **4. Run the Application**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

## 🔄 Integration with Existing System

### **Automatic Medical Record Creation**
When a patient uploads an image for analysis, the system automatically:
1. Processes the image with Gemini AI
2. Creates a medical record with the analysis results
3. Links the record to the patient's profile
4. Stores the medical image securely

### **Enhanced Appointment System**
The existing appointment system now:
1. Links appointments to patient records
2. Tracks patient symptoms and history
3. Provides comprehensive patient context for doctors

### **Seamless Migration**
- Existing functionality remains unchanged
- New features are additive
- Gradual migration path available

## 📊 Features

### **Patient Management**
- ✅ Create, read, update, delete patients
- ✅ Search patients by multiple criteria
- ✅ Comprehensive patient profiles
- ✅ Allergy and medical history tracking

### **Medical Records**
- ✅ Full CRUD operations for medical records
- ✅ Image storage and retrieval
- ✅ AI analysis integration
- ✅ Condition history tracking
- ✅ Timeline and filtering capabilities

### **Data Analytics**
- ✅ Patient health statistics
- ✅ Medical record summaries
- ✅ Condition frequency analysis
- ✅ Trend identification

### **Security & Privacy**
- ✅ Data validation with Pydantic
- ✅ Secure file storage
- ✅ Access control (ready for implementation)
- ✅ Audit trails

## 🔮 Future Enhancements

### **Phase 2: Advanced Features**
- [ ] Patient authentication and login
- [ ] Role-based access control
- [ ] HIPAA compliance features
- [ ] Data encryption at rest
- [ ] Backup and recovery systems

### **Phase 3: Analytics & AI**
- [ ] Predictive health analytics
- [ ] Disease progression tracking
- [ ] Treatment effectiveness analysis
- [ ] Population health insights

### **Phase 4: Integration**
- [ ] Electronic Health Record (EHR) integration
- [ ] Lab result integration
- [ ] Pharmacy integration
- [ ] Insurance system integration

## 🧪 Testing

### **API Testing**
```bash
# Test patient creation
curl -X POST "http://localhost:8001/api/patients/" \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Patient","email":"test@example.com"}'

# Test patient retrieval
curl "http://localhost:8001/api/patients/email/test@example.com"

# Test medical record creation
curl -X POST "http://localhost:8001/api/patients/{patient_id}/medical-records/" \
  -H "Content-Type: application/json" \
  -d '{"record_type":"xray","modality":"xray","diagnosis":"Normal"}'
```

### **Database Testing**
```bash
# Check SQLite database
sqlite3 maruthuvam_ai.db ".tables"
sqlite3 maruthuvam_ai.db "SELECT * FROM patients;"
```

## 📝 Usage Examples

### **1. Complete Patient Workflow**
```python
# 1. Create patient
patient_data = {
    "name": "Jane Smith",
    "email": "jane.smith@example.com",
    "phone": "+91-98765-43211",
    "date_of_birth": "1985-05-15",
    "gender": "female",
    "blood_type": "A+"
}

# 2. Upload X-ray image
# 3. System automatically creates medical record
# 4. View patient summary with all records
# 5. Track condition history over time
```

### **2. Medical Record Management**
```python
# Get patient's complete medical history
medical_history = await get_medical_history(patient_id)

# Filter by modality
xray_records = await get_records_by_modality(patient_id, "xray")

# Get condition timeline
pneumonia_history = await get_condition_history(patient_id, "pneumonia")
```

## 🚨 Error Handling

The system includes comprehensive error handling:
- **Validation Errors**: Pydantic model validation
- **Database Errors**: Connection and query error handling
- **File Errors**: Upload and storage error handling
- **API Errors**: HTTP status codes and error messages

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## 🤝 Contributing

1. Follow the modular architecture pattern
2. Add comprehensive error handling
3. Include Pydantic models for all data
4. Write tests for new features
5. Update documentation

## 📄 License

This module is part of the Maruthuvam AI platform and follows the same MIT license.

---

**Built with ❤️ for better healthcare management** 