from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class LogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"

class ActivityType(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    IMAGE_UPLOAD = "image_upload"
    ANALYSIS_REQUEST = "analysis_request"
    REPORT_GENERATION = "report_generation"
    APPOINTMENT_BOOKING = "appointment_booking"
    PATIENT_CREATION = "patient_creation"
    MEDICAL_RECORD_CREATION = "medical_record_creation"
    ADMIN_ACTION = "admin_action"

class ModerationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"

class UserActivityLog(BaseModel):
    id: str = Field(..., description="Unique log ID")
    user_id: Optional[str] = Field(None, description="User ID if authenticated")
    user_email: Optional[str] = Field(None, description="User email")
    activity_type: ActivityType = Field(..., description="Type of activity")
    description: str = Field(..., description="Activity description")
    ip_address: Optional[str] = Field(None, description="User IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(..., description="Activity timestamp")
    session_id: Optional[str] = Field(None, description="Session ID")

class SystemLog(BaseModel):
    id: str = Field(..., description="Unique log ID")
    level: LogLevel = Field(..., description="Log level")
    component: str = Field(..., description="System component")
    message: str = Field(..., description="Log message")
    stack_trace: Optional[str] = Field(None, description="Stack trace for errors")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(..., description="Log timestamp")

class AnalyticsData(BaseModel):
    total_users: int = Field(..., description="Total registered users")
    active_users_today: int = Field(..., description="Active users today")
    total_analyses: int = Field(..., description="Total image analyses")
    analyses_today: int = Field(..., description="Analyses performed today")
    total_appointments: int = Field(..., description="Total appointments")
    appointments_today: int = Field(..., description="Appointments today")
    total_patients: int = Field(..., description="Total patients")
    patients_today: int = Field(..., description="New patients today")
    system_uptime: float = Field(..., description="System uptime in hours")
    average_response_time: float = Field(..., description="Average API response time")
    error_rate: float = Field(..., description="Error rate percentage")
    gemini_api_calls: int = Field(..., description="Total Gemini API calls")
    gemini_api_calls_today: int = Field(..., description="Gemini API calls today")

class ModerationAction(BaseModel):
    id: str = Field(..., description="Unique action ID")
    admin_id: str = Field(..., description="Admin who performed the action")
    admin_email: str = Field(..., description="Admin email")
    target_type: str = Field(..., description="Type of content being moderated")
    target_id: str = Field(..., description="ID of the target content")
    action_type: str = Field(..., description="Type of moderation action")
    reason: Optional[str] = Field(None, description="Reason for the action")
    status: ModerationStatus = Field(..., description="Moderation status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(..., description="Action timestamp")

class ContentFlag(BaseModel):
    id: str = Field(..., description="Unique flag ID")
    content_type: str = Field(..., description="Type of flagged content")
    content_id: str = Field(..., description="ID of flagged content")
    reporter_id: Optional[str] = Field(None, description="User who reported the content")
    reporter_email: Optional[str] = Field(None, description="Reporter email")
    reason: str = Field(..., description="Reason for flagging")
    description: Optional[str] = Field(None, description="Additional description")
    status: ModerationStatus = Field(default=ModerationStatus.PENDING, description="Flag status")
    admin_notes: Optional[str] = Field(None, description="Admin notes")
    timestamp: datetime = Field(..., description="Flag timestamp")

class AdminUser(BaseModel):
    id: str = Field(..., description="Admin user ID")
    email: str = Field(..., description="Admin email")
    name: str = Field(..., description="Admin name")
    role: str = Field(..., description="Admin role")
    permissions: List[str] = Field(..., description="Admin permissions")
    is_active: bool = Field(..., description="Whether admin is active")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    created_at: datetime = Field(..., description="Account creation timestamp")

class DashboardStats(BaseModel):
    analytics: AnalyticsData = Field(..., description="System analytics")
    recent_activities: List[UserActivityLog] = Field(..., description="Recent user activities")
    recent_logs: List[SystemLog] = Field(..., description="Recent system logs")
    pending_flags: List[ContentFlag] = Field(..., description="Pending content flags")
    pending_moderations: List[ModerationAction] = Field(..., description="Pending moderation actions")

class LogFilter(BaseModel):
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(None, description="End date for filtering")
    level: Optional[LogLevel] = Field(None, description="Log level filter")
    component: Optional[str] = Field(None, description="Component filter")
    user_id: Optional[str] = Field(None, description="User ID filter")
    activity_type: Optional[ActivityType] = Field(None, description="Activity type filter")
    limit: int = Field(default=100, description="Number of logs to return")

class AnalyticsFilter(BaseModel):
    start_date: Optional[datetime] = Field(None, description="Start date for analytics")
    end_date: Optional[datetime] = Field(None, description="End date for analytics")
    group_by: Optional[str] = Field(None, description="Group by field (day, hour, etc.)") 