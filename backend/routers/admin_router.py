from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

from services.admin_service import AdminService
from models.admin_models import (
    LogFilter, AnalyticsFilter, ContentFlag, ModerationAction,
    ActivityType, LogLevel, ModerationStatus
)

router = APIRouter(prefix="/api/admin", tags=["Admin Dashboard"])

# Global admin service instance
_admin_service = None

# Dependency to get admin service
async def get_admin_service():
    global _admin_service
    if _admin_service is None:
        _admin_service = AdminService()
        await _admin_service.initialize()
    return _admin_service

# Dependency to check admin authentication (placeholder for now)
async def verify_admin_access(request: Request):
    # In production, implement proper admin authentication
    # For now, we'll allow access to all admin endpoints
    return True

@router.get("/dashboard")
async def get_dashboard_stats(
    service: AdminService = Depends(get_admin_service),
    _: bool = Depends(verify_admin_access)
):
    """Get comprehensive dashboard statistics"""
    try:
        stats = await service.get_dashboard_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard stats: {str(e)}")

@router.get("/analytics")
async def get_analytics(
    start_date: Optional[datetime] = Query(None, description="Start date for analytics"),
    end_date: Optional[datetime] = Query(None, description="End date for analytics"),
    group_by: Optional[str] = Query(None, description="Group by field (day, hour, etc.)"),
    service: AdminService = Depends(get_admin_service),
    _: bool = Depends(verify_admin_access)
):
    """Get system analytics data"""
    try:
        filter_params = AnalyticsFilter(
            start_date=start_date,
            end_date=end_date,
            group_by=group_by
        )
        analytics = await service.get_analytics(filter_params)
        return JSONResponse(content=analytics.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")

@router.get("/logs/user-activities")
async def get_user_activities(
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    level: Optional[LogLevel] = Query(None, description="Log level filter"),
    component: Optional[str] = Query(None, description="Component filter"),
    user_id: Optional[str] = Query(None, description="User ID filter"),
    activity_type: Optional[ActivityType] = Query(None, description="Activity type filter"),
    limit: int = Query(100, description="Number of logs to return"),
    service: AdminService = Depends(get_admin_service),
    _: bool = Depends(verify_admin_access)
):
    """Get user activity logs with filtering"""
    try:
        filter_params = LogFilter(
            start_date=start_date,
            end_date=end_date,
            level=level,
            component=component,
            user_id=user_id,
            activity_type=activity_type,
            limit=limit
        )
        activities = await service.get_user_activities(filter_params)
        return JSONResponse(content=[activity.dict() for activity in activities])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user activities: {str(e)}")

@router.get("/logs/system")
async def get_system_logs(
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    level: Optional[LogLevel] = Query(None, description="Log level filter"),
    component: Optional[str] = Query(None, description="Component filter"),
    limit: int = Query(100, description="Number of logs to return"),
    service: AdminService = Depends(get_admin_service),
    _: bool = Depends(verify_admin_access)
):
    """Get system logs with filtering"""
    try:
        filter_params = LogFilter(
            start_date=start_date,
            end_date=end_date,
            level=level,
            component=component,
            limit=limit
        )
        logs = await service.get_system_logs(filter_params)
        return JSONResponse(content=[log.dict() for log in logs])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching system logs: {str(e)}")

@router.get("/moderation/flags")
async def get_pending_flags(
    service: AdminService = Depends(get_admin_service),
    _: bool = Depends(verify_admin_access)
):
    """Get pending content flags for moderation"""
    try:
        flags = await service.get_pending_content_flags()
        return JSONResponse(content=[flag.dict() for flag in flags])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching pending flags: {str(e)}")

@router.post("/moderation/flags")
async def create_content_flag(
    content_type: str,
    content_id: str,
    reason: str,
    reporter_id: Optional[str] = None,
    reporter_email: Optional[str] = None,
    description: Optional[str] = None,
    service: AdminService = Depends(get_admin_service)
):
    """Create a new content flag"""
    try:
        flag = await service.create_content_flag(
            content_type=content_type,
            content_id=content_id,
            reason=reason,
            reporter_id=reporter_id,
            reporter_email=reporter_email,
            description=description
        )
        if flag:
            return JSONResponse(content=flag.dict(), status_code=201)
        else:
            raise HTTPException(status_code=400, detail="Failed to create content flag")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating content flag: {str(e)}")

@router.put("/moderation/flags/{flag_id}")
async def moderate_content(
    flag_id: str,
    action: str,
    status: ModerationStatus,
    admin_id: str,
    admin_email: str,
    reason: Optional[str] = None,
    admin_notes: Optional[str] = None,
    service: AdminService = Depends(get_admin_service),
    _: bool = Depends(verify_admin_access)
):
    """Moderate flagged content"""
    try:
        success = await service.moderate_content(
            flag_id=flag_id,
            admin_id=admin_id,
            admin_email=admin_email,
            action=action,
            status=status,
            reason=reason,
            admin_notes=admin_notes
        )
        if success:
            return JSONResponse(content={"message": "Content moderated successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to moderate content")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error moderating content: {str(e)}")

@router.get("/health")
async def get_system_health(
    service: AdminService = Depends(get_admin_service),
    _: bool = Depends(verify_admin_access)
):
    """Get system health metrics"""
    try:
        health = await service.get_system_health()
        return JSONResponse(content=health)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching system health: {str(e)}")

@router.post("/logs/activity")
async def log_user_activity(
    activity_type: ActivityType,
    description: str,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    service: AdminService = Depends(get_admin_service)
):
    """Log user activity (for internal use)"""
    try:
        success = await service.log_user_activity(
            activity_type=activity_type,
            description=description,
            user_id=user_id,
            user_email=user_email,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata,
            session_id=session_id
        )
        if success:
            return JSONResponse(content={"message": "Activity logged successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to log activity")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging activity: {str(e)}")

@router.post("/logs/system")
async def log_system_event(
    level: LogLevel,
    component: str,
    message: str,
    stack_trace: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    service: AdminService = Depends(get_admin_service)
):
    """Log system event (for internal use)"""
    try:
        success = await service.log_system_event(
            level=level,
            component=component,
            message=message,
            stack_trace=stack_trace,
            metadata=metadata
        )
        if success:
            return JSONResponse(content={"message": "System event logged successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to log system event")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging system event: {str(e)}")

@router.get("/stats/realtime")
async def get_realtime_stats(
    service: AdminService = Depends(get_admin_service),
    _: bool = Depends(verify_admin_access)
):
    """Get real-time system statistics"""
    try:
        # Get current analytics
        analytics = await service.get_analytics()
        
        # Get recent activities (last 5 minutes)
        recent_activities = await service.get_user_activities()
        recent_activities = [a for a in recent_activities 
                           if a.timestamp > datetime.now() - timedelta(minutes=5)]
        
        # Get recent system logs (last 5 minutes)
        recent_logs = await service.get_system_logs()
        recent_logs = [l for l in recent_logs 
                      if l.timestamp > datetime.now() - timedelta(minutes=5)]
        
        return JSONResponse(content={
            "timestamp": datetime.now().isoformat(),
            "analytics": analytics.dict(),
            "recent_activities_count": len(recent_activities),
            "recent_logs_count": len(recent_logs),
            "error_logs_count": len([l for l in recent_logs if l.level == LogLevel.ERROR])
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching real-time stats: {str(e)}")

@router.get("/export/logs")
async def export_logs(
    log_type: str = Query(..., description="Type of logs to export (user_activities, system)"),
    start_date: Optional[datetime] = Query(None, description="Start date for export"),
    end_date: Optional[datetime] = Query(None, description="End date for export"),
    format: str = Query("json", description="Export format (json, csv)"),
    service: AdminService = Depends(get_admin_service),
    _: bool = Depends(verify_admin_access)
):
    """Export logs in specified format"""
    try:
        if log_type == "user_activities":
            logs = await service.get_user_activities()
        elif log_type == "system":
            logs = await service.get_system_logs()
        else:
            raise HTTPException(status_code=400, detail="Invalid log type")
        
        # Filter by date if provided
        if start_date:
            logs = [log for log in logs if log.timestamp >= start_date]
        if end_date:
            logs = [log for log in logs if log.timestamp <= end_date]
        
        if format.lower() == "json":
            return JSONResponse(content=[log.dict() for log in logs])
        elif format.lower() == "csv":
            # Convert to CSV format
            if not logs:
                return JSONResponse(content="No logs found")
            
            # Get headers from first log
            headers = list(logs[0].dict().keys())
            csv_content = ",".join(headers) + "\n"
            
            for log in logs:
                values = [str(log.dict()[header]) for header in headers]
                csv_content += ",".join(values) + "\n"
            
            return JSONResponse(content=csv_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting logs: {str(e)}") 