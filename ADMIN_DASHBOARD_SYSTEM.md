# 🛡️ Admin Dashboard System - Maruthuvam AI

## 📋 Overview

The Admin Dashboard System is a comprehensive monitoring and management platform that provides administrators with real-time insights into system performance, user activities, and content moderation capabilities. It offers a modular, scalable architecture with advanced analytics and logging features.

## 🏗️ Architecture

### **Modular Design Pattern**
```
backend/
├── models/
│   └── admin_models.py          # Pydantic models for admin data
├── database/
│   └── admin_manager.py         # Admin-specific database operations
├── services/
│   └── admin_service.py         # Business logic for admin operations
├── routers/
│   └── admin_router.py          # Admin API endpoints
└── main.py                      # Integration with main app

frontend/
└── src/
    └── components/
        └── AdminDashboard.jsx   # Main admin dashboard UI
```

### **Key Components**

1. **Models Layer**: Pydantic models for data validation and serialization
2. **Database Layer**: Admin-specific database operations with SQLite/PostgreSQL support
3. **Service Layer**: Business logic for logging, analytics, and moderation
4. **Router Layer**: FastAPI endpoints for admin functionality
5. **Frontend Layer**: React-based admin dashboard with real-time updates

## 🗄️ Database Schema

### **User Activity Logs Table**
```sql
CREATE TABLE user_activity_logs (
    id TEXT PRIMARY KEY,           -- UUID
    user_id TEXT,                  -- User ID if authenticated
    user_email TEXT,               -- User email
    activity_type TEXT NOT NULL,   -- Type of activity
    description TEXT NOT NULL,     -- Activity description
    ip_address TEXT,               -- User IP address
    user_agent TEXT,               -- User agent string
    metadata TEXT,                 -- JSON metadata
    timestamp TEXT NOT NULL,       -- Activity timestamp
    session_id TEXT                -- Session ID
);
```

### **System Logs Table**
```sql
CREATE TABLE system_logs (
    id TEXT PRIMARY KEY,           -- UUID
    level TEXT NOT NULL,           -- Log level (info, warning, error, debug)
    component TEXT NOT NULL,       -- System component
    message TEXT NOT NULL,         -- Log message
    stack_trace TEXT,              -- Stack trace for errors
    metadata TEXT,                 -- JSON metadata
    timestamp TEXT NOT NULL        -- Log timestamp
);
```

### **Admin Users Table**
```sql
CREATE TABLE admin_users (
    id TEXT PRIMARY KEY,           -- UUID
    email TEXT UNIQUE NOT NULL,    -- Admin email
    name TEXT NOT NULL,            -- Admin name
    role TEXT NOT NULL,            -- Admin role
    permissions TEXT NOT NULL,     -- JSON permissions array
    is_active BOOLEAN NOT NULL,    -- Whether admin is active
    last_login TEXT,               -- Last login timestamp
    created_at TEXT NOT NULL       -- Account creation timestamp
);
```

### **Moderation Actions Table**
```sql
CREATE TABLE moderation_actions (
    id TEXT PRIMARY KEY,           -- UUID
    admin_id TEXT NOT NULL,        -- Admin who performed action
    admin_email TEXT NOT NULL,     -- Admin email
    target_type TEXT NOT NULL,     -- Type of content being moderated
    target_id TEXT NOT NULL,       -- ID of target content
    action_type TEXT NOT NULL,     -- Type of moderation action
    reason TEXT,                   -- Reason for action
    status TEXT NOT NULL,          -- Moderation status
    metadata TEXT,                 -- JSON metadata
    timestamp TEXT NOT NULL        -- Action timestamp
);
```

### **Content Flags Table**
```sql
CREATE TABLE content_flags (
    id TEXT PRIMARY KEY,           -- UUID
    content_type TEXT NOT NULL,    -- Type of flagged content
    content_id TEXT NOT NULL,      -- ID of flagged content
    reporter_id TEXT,              -- User who reported content
    reporter_email TEXT,           -- Reporter email
    reason TEXT NOT NULL,          -- Reason for flagging
    description TEXT,              -- Additional description
    status TEXT NOT NULL,          -- Flag status (pending, approved, rejected)
    admin_notes TEXT,              -- Admin notes
    timestamp TEXT NOT NULL        -- Flag timestamp
);
```

### **Analytics Cache Table**
```sql
CREATE TABLE analytics_cache (
    id TEXT PRIMARY KEY,           -- UUID
    cache_key TEXT UNIQUE NOT NULL, -- Cache key
    data TEXT NOT NULL,            -- Cached data (JSON)
    expires_at TEXT NOT NULL,      -- Cache expiration
    created_at TEXT NOT NULL       -- Cache creation timestamp
);
```

## 🔌 API Endpoints

### **Dashboard & Analytics**

#### `GET /api/admin/dashboard`
Get comprehensive dashboard statistics
```json
{
  "analytics": {
    "total_users": 150,
    "active_users_today": 25,
    "total_analyses": 500,
    "analyses_today": 15,
    "total_appointments": 75,
    "appointments_today": 8,
    "total_patients": 120,
    "patients_today": 12,
    "system_uptime": 168.5,
    "average_response_time": 0.5,
    "error_rate": 2.1,
    "gemini_api_calls": 1250,
    "gemini_api_calls_today": 45
  },
  "recent_activities": [...],
  "recent_logs": [...],
  "pending_flags": [...],
  "system_info": {
    "uptime_hours": 168.5,
    "start_time": "2024-01-01T00:00:00",
    "current_time": "2024-01-08T12:00:00"
  }
}
```

#### `GET /api/admin/analytics`
Get system analytics with filtering
```json
{
  "total_users": 150,
  "active_users_today": 25,
  "total_analyses": 500,
  "analyses_today": 15,
  "total_appointments": 75,
  "appointments_today": 8,
  "total_patients": 120,
  "patients_today": 12,
  "system_uptime": 168.5,
  "average_response_time": 0.5,
  "error_rate": 2.1,
  "gemini_api_calls": 1250,
  "gemini_api_calls_today": 45
}
```

#### `GET /api/admin/stats/realtime`
Get real-time system statistics
```json
{
  "timestamp": "2024-01-08T12:00:00",
  "analytics": {...},
  "recent_activities_count": 5,
  "recent_logs_count": 3,
  "error_logs_count": 1
}
```

#### `GET /api/admin/health`
Get system health metrics
```json
{
  "health_score": 85.5,
  "status": "healthy",
  "metrics": {
    "error_rate": 2.1,
    "response_time": 0.5,
    "uptime": 168.5,
    "active_users": 25
  },
  "last_updated": "2024-01-08T12:00:00"
}
```

### **Logs Management**

#### `GET /api/admin/logs/user-activities`
Get user activity logs with filtering
```json
[
  {
    "id": "uuid",
    "user_id": "user123",
    "user_email": "user@example.com",
    "activity_type": "image_upload",
    "description": "X-ray image uploaded for analysis",
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "metadata": {"file_name": "xray.png", "file_size": 1024000},
    "timestamp": "2024-01-08T12:00:00",
    "session_id": "session123"
  }
]
```

#### `GET /api/admin/logs/system`
Get system logs with filtering
```json
[
  {
    "id": "uuid",
    "level": "info",
    "component": "xray_prediction",
    "message": "X-ray analysis completed successfully",
    "stack_trace": null,
    "metadata": {"conditions_found": 2, "confidence": 0.85},
    "timestamp": "2024-01-08T12:00:00"
  }
]
```

#### `GET /api/admin/export/logs`
Export logs in JSON or CSV format
- Query parameters: `log_type`, `start_date`, `end_date`, `format`

### **Content Moderation**

#### `GET /api/admin/moderation/flags`
Get pending content flags
```json
[
  {
    "id": "uuid",
    "content_type": "medical_image",
    "content_id": "image123",
    "reporter_id": "user456",
    "reporter_email": "reporter@example.com",
    "reason": "Inappropriate content",
    "description": "Image contains sensitive information",
    "status": "pending",
    "admin_notes": null,
    "timestamp": "2024-01-08T12:00:00"
  }
]
```

#### `POST /api/admin/moderation/flags`
Create a new content flag
```json
{
  "content_type": "medical_image",
  "content_id": "image123",
  "reason": "Inappropriate content",
  "reporter_id": "user456",
  "reporter_email": "reporter@example.com",
  "description": "Image contains sensitive information"
}
```

#### `PUT /api/admin/moderation/flags/{flag_id}`
Moderate flagged content
```json
{
  "action": "approve",
  "status": "approved",
  "admin_id": "admin123",
  "admin_email": "admin@example.com",
  "reason": "Content is appropriate",
  "admin_notes": "Reviewed and approved"
}
```

### **Internal Logging Endpoints**

#### `POST /api/admin/logs/activity`
Log user activity (internal use)
```json
{
  "activity_type": "image_upload",
  "description": "X-ray image uploaded for analysis",
  "user_id": "user123",
  "user_email": "user@example.com",
  "ip_address": "192.168.1.1",
  "user_agent": "Mozilla/5.0...",
  "metadata": {"file_name": "xray.png"},
  "session_id": "session123"
}
```

#### `POST /api/admin/logs/system`
Log system event (internal use)
```json
{
  "level": "info",
  "component": "xray_prediction",
  "message": "X-ray analysis completed successfully",
  "stack_trace": null,
  "metadata": {"conditions_found": 2}
}
```

## 🚀 Setup and Configuration

### **1. Environment Variables**
Add to your `.env` file:
```bash
# Admin Dashboard Configuration
ADMIN_DASHBOARD_ENABLED=true
ADMIN_LOG_RETENTION_DAYS=30
ADMIN_MAX_LOG_ENTRIES=10000
ADMIN_REALTIME_UPDATE_INTERVAL=30000

# Admin Authentication (for production)
ADMIN_SECRET_KEY=your-admin-secret-key
ADMIN_SESSION_TIMEOUT=3600
```

### **2. Database Setup**
The admin tables are automatically created when the service initializes:
```bash
# The system will create all necessary tables on first run
# No manual setup required
```

### **3. Frontend Integration**
The admin dashboard is automatically integrated into the main application:
- Route: `/admin`
- Navigation: Added to header with shield icon
- Accessible from any authenticated session

## 📊 Features

### **Dashboard Overview**
- ✅ Real-time system health monitoring
- ✅ Key performance indicators (KPIs)
- ✅ User activity tracking
- ✅ System performance metrics
- ✅ Error rate monitoring
- ✅ API usage statistics

### **Analytics & Reporting**
- ✅ Comprehensive analytics data
- ✅ Real-time statistics updates
- ✅ Historical data tracking
- ✅ Performance trend analysis
- ✅ Export functionality (JSON/CSV)
- ✅ Custom date range filtering

### **Log Management**
- ✅ User activity logging
- ✅ System event logging
- ✅ Error tracking and monitoring
- ✅ Log filtering and search
- ✅ Log export capabilities
- ✅ Log retention management

### **Content Moderation**
- ✅ Content flag management
- ✅ Moderation workflow
- ✅ Admin action tracking
- ✅ Content review system
- ✅ Moderation history
- ✅ Bulk moderation actions

### **System Monitoring**
- ✅ System health scoring
- ✅ Performance monitoring
- ✅ Error rate tracking
- ✅ Uptime monitoring
- ✅ Response time analysis
- ✅ Resource usage tracking

## 🎨 Frontend Features

### **Dashboard Interface**
- **Overview Tab**: Key metrics and recent activities
- **Logs Tab**: User activity and system logs
- **Moderation Tab**: Content flag management
- **Settings Tab**: Admin configuration

### **Real-time Updates**
- Auto-refresh every 30 seconds
- Live system health monitoring
- Real-time activity feed
- Instant notification updates

### **Responsive Design**
- Mobile-friendly interface
- Adaptive layout
- Touch-optimized controls
- Cross-browser compatibility

### **Interactive Elements**
- Tabbed navigation
- Progress indicators
- Status badges
- Action buttons
- Filter controls

## 🔧 Integration with Existing System

### **Automatic Logging**
The admin system automatically logs:
- User image uploads
- Analysis requests
- Medical record creation
- Appointment bookings
- System errors and warnings
- API usage statistics

### **Enhanced Monitoring**
- Integration with existing endpoints
- Automatic error tracking
- Performance monitoring
- User activity tracking
- System health monitoring

### **Seamless Integration**
- No changes to existing functionality
- Additive features only
- Backward compatibility
- Gradual rollout capability

## 🧪 Testing

### **API Testing**
```bash
# Test dashboard endpoint
curl -X GET "http://localhost:8001/api/admin/dashboard"

# Test analytics endpoint
curl -X GET "http://localhost:8001/api/admin/analytics"

# Test health endpoint
curl -X GET "http://localhost:8001/api/admin/health"

# Test logs endpoint
curl -X GET "http://localhost:8001/api/admin/logs/user-activities"

# Test moderation endpoint
curl -X GET "http://localhost:8001/api/admin/moderation/flags"
```

### **Frontend Testing**
1. Navigate to `/admin` in the browser
2. Verify all tabs load correctly
3. Check real-time updates
4. Test moderation actions
5. Verify log export functionality

## 🔮 Future Enhancements

### **Phase 2: Advanced Analytics**
- [ ] Custom dashboard widgets
- [ ] Advanced filtering options
- [ ] Data visualization charts
- [ ] Predictive analytics
- [ ] Machine learning insights

### **Phase 3: Enhanced Moderation**
- [ ] AI-powered content analysis
- [ ] Automated moderation rules
- [ ] Bulk moderation tools
- [ ] Moderation queue management
- [ ] Content quality scoring

### **Phase 4: Advanced Monitoring**
- [ ] Alert system integration
- [ ] Email notifications
- [ ] Slack/Discord integration
- [ ] Performance benchmarking
- [ ] Capacity planning tools

### **Phase 5: Security & Compliance**
- [ ] Role-based access control
- [ ] Audit trail management
- [ ] GDPR compliance features
- [ ] Data encryption
- [ ] Security monitoring

## 📝 Usage Examples

### **1. Monitoring System Health**
```javascript
// Check system health
const health = await fetch('/api/admin/health');
const healthData = await health.json();

if (healthData.status === 'critical') {
  // Send alert to administrators
  sendAlert('System health is critical');
}
```

### **2. Tracking User Activities**
```javascript
// Log user activity
await fetch('/api/admin/logs/activity', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    activity_type: 'image_upload',
    description: 'User uploaded X-ray image',
    user_email: 'user@example.com'
  })
});
```

### **3. Content Moderation**
```javascript
// Moderate flagged content
await fetch(`/api/admin/moderation/flags/${flagId}`, {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    action: 'approve',
    status: 'approved',
    admin_id: 'admin123',
    admin_email: 'admin@example.com'
  })
});
```

## 🚨 Error Handling

The admin system includes comprehensive error handling:
- **API Errors**: Proper HTTP status codes and error messages
- **Database Errors**: Connection and query error handling
- **Validation Errors**: Pydantic model validation
- **Frontend Errors**: User-friendly error messages
- **Logging Errors**: Graceful degradation

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
6. Follow security best practices

## 📄 License

This module is part of the Maruthuvam AI platform and follows the same MIT license.

---

**Built with ❤️ for comprehensive system monitoring and management** 