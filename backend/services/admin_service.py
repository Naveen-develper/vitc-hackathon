from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import json
from database.config import DatabaseConfig
from database.admin_manager import AdminDatabaseManager
from models.admin_models import (
    UserActivityLog, SystemLog, AnalyticsData, ModerationAction, 
    ContentFlag, AdminUser, LogFilter, AnalyticsFilter, ActivityType, LogLevel
)

class AdminService:
    """Service layer for admin operations"""
    
    def __init__(self):
        self.base_db = DatabaseConfig.get_database_manager()
        self.admin_db = AdminDatabaseManager(self.base_db)
        self.start_time = datetime.now()
    
    async def initialize(self):
        """Initialize database connection and create admin tables"""
        try:
            # Ensure base database is connected
            if not hasattr(self.base_db, 'connection') and not hasattr(self.base_db, 'db'):
                await self.base_db.connect()
            
            await self.base_db.create_tables()
            
            # Reinitialize admin_db with the connected base_db
            self.admin_db = AdminDatabaseManager(self.base_db)
            await self.admin_db.create_admin_tables()
            
            # Log system startup
            await self.log_system_event(
                LogLevel.INFO,
                "system",
                "Admin service initialized successfully",
                metadata={"start_time": self.start_time.isoformat()}
            )
            return True
        except Exception as e:
            print(f"Error initializing admin service: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup database connection"""
        try:
            await self.base_db.disconnect()
            return True
        except Exception as e:
            print(f"Error cleaning up admin service: {e}")
            return False
    
    async def log_user_activity(
        self,
        activity_type: ActivityType,
        description: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> bool:
        """Log user activity"""
        try:
            activity = UserActivityLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                user_email=user_email,
                activity_type=activity_type,
                description=description,
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata,
                timestamp=datetime.now(),
                session_id=session_id
            )
            return await self.admin_db.log_user_activity(activity)
        except Exception as e:
            print(f"Error logging user activity: {e}")
            return False
    
    async def log_system_event(
        self,
        level: LogLevel,
        component: str,
        message: str,
        stack_trace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log system event"""
        try:
            log = SystemLog(
                id=str(uuid.uuid4()),
                level=level,
                component=component,
                message=message,
                stack_trace=stack_trace,
                metadata=metadata,
                timestamp=datetime.now()
            )
            return await self.admin_db.log_system_event(log)
        except Exception as e:
            print(f"Error logging system event: {e}")
            return False
    
    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get comprehensive dashboard statistics"""
        try:
            analytics = await self.admin_db.get_analytics_data()
            recent_activities = await self.admin_db.get_recent_activities(10)
            recent_logs = await self.admin_db.get_recent_logs(10)
            pending_flags = await self.admin_db.get_pending_flags(10)
            
            return {
                "analytics": analytics.dict(),
                "recent_activities": [activity.dict() for activity in recent_activities],
                "recent_logs": [log.dict() for log in recent_logs],
                "pending_flags": [flag.dict() for flag in pending_flags],
                "system_info": {
                    "uptime_hours": analytics.system_uptime,
                    "start_time": self.start_time.isoformat(),
                    "current_time": datetime.now().isoformat()
                }
            }
        except Exception as e:
            print(f"Error getting dashboard stats: {e}")
            return {
                "analytics": {},
                "recent_activities": [],
                "recent_logs": [],
                "pending_flags": [],
                "system_info": {}
            }
    
    async def get_analytics(self, filter_params: AnalyticsFilter = None) -> AnalyticsData:
        """Get analytics data with optional filtering"""
        try:
            return await self.admin_db.get_analytics_data(filter_params)
        except Exception as e:
            print(f"Error getting analytics: {e}")
            return AnalyticsData(
                total_users=0, active_users_today=0, total_analyses=0, analyses_today=0,
                total_appointments=0, appointments_today=0, total_patients=0, patients_today=0,
                system_uptime=0.0, average_response_time=0.0, error_rate=0.0,
                gemini_api_calls=0, gemini_api_calls_today=0
            )
    
    async def get_user_activities(self, filter_params: LogFilter = None) -> List[UserActivityLog]:
        """Get user activities with optional filtering"""
        try:
            # For now, return recent activities. In production, implement proper filtering
            limit = filter_params.limit if filter_params else 100
            return await self.admin_db.get_recent_activities(limit)
        except Exception as e:
            print(f"Error getting user activities: {e}")
            return []
    
    async def get_system_logs(self, filter_params: LogFilter = None) -> List[SystemLog]:
        """Get system logs with optional filtering"""
        try:
            # For now, return recent logs. In production, implement proper filtering
            limit = filter_params.limit if filter_params else 100
            return await self.admin_db.get_recent_logs(limit)
        except Exception as e:
            print(f"Error getting system logs: {e}")
            return []
    
    async def get_pending_content_flags(self) -> List[ContentFlag]:
        """Get pending content flags for moderation"""
        try:
            return await self.admin_db.get_pending_flags(50)
        except Exception as e:
            print(f"Error getting pending flags: {e}")
            return []
    
    async def create_content_flag(
        self,
        content_type: str,
        content_id: str,
        reason: str,
        reporter_id: Optional[str] = None,
        reporter_email: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[ContentFlag]:
        """Create a new content flag"""
        try:
            flag = ContentFlag(
                id=str(uuid.uuid4()),
                content_type=content_type,
                content_id=content_id,
                reporter_id=reporter_id,
                reporter_email=reporter_email,
                reason=reason,
                description=description,
                status="pending",
                timestamp=datetime.now()
            )
            
            # Store in database
            success = await self._store_content_flag(flag)
            if success:
                # Log the flag creation
                await self.log_system_event(
                    LogLevel.INFO,
                    "moderation",
                    f"Content flag created for {content_type}:{content_id}",
                    metadata={"flag_id": flag.id, "reason": reason}
                )
                return flag
            return None
        except Exception as e:
            print(f"Error creating content flag: {e}")
            return None
    
    async def _store_content_flag(self, flag: ContentFlag) -> bool:
        """Store content flag in database"""
        try:
            if hasattr(self.admin_db.base_manager, 'db'):
                db = self.admin_db.base_manager.db
                if hasattr(db, 'cursor'):  # SQLite
                    cursor = db.cursor()
                    cursor.execute('''
                        INSERT INTO content_flags 
                        (id, content_type, content_id, reporter_id, reporter_email, reason, description, status, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        flag.id, flag.content_type, flag.content_id, flag.reporter_id,
                        flag.reporter_email, flag.reason, flag.description, flag.status,
                        flag.timestamp.isoformat()
                    ))
                    db.commit()
                else:  # PostgreSQL
                    await db.execute('''
                        INSERT INTO content_flags 
                        (id, content_type, content_id, reporter_id, reporter_email, reason, description, status, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ''', (
                        flag.id, flag.content_type, flag.content_id, flag.reporter_id,
                        flag.reporter_email, flag.reason, flag.description, flag.status,
                        flag.timestamp
                    ))
                return True
        except Exception as e:
            print(f"Error storing content flag: {e}")
            return False
    
    async def moderate_content(
        self,
        flag_id: str,
        admin_id: str,
        admin_email: str,
        action: str,
        status: str,
        reason: Optional[str] = None,
        admin_notes: Optional[str] = None
    ) -> bool:
        """Moderate flagged content"""
        try:
            # Update flag status
            success = await self._update_flag_status(flag_id, status, admin_notes)
            if not success:
                return False
            
            # Create moderation action record
            action_record = ModerationAction(
                id=str(uuid.uuid4()),
                admin_id=admin_id,
                admin_email=admin_email,
                target_type="content_flag",
                target_id=flag_id,
                action_type=action,
                reason=reason,
                status=status,
                timestamp=datetime.now()
            )
            
            success = await self._store_moderation_action(action_record)
            if success:
                # Log the moderation action
                await self.log_system_event(
                    LogLevel.INFO,
                    "moderation",
                    f"Content moderated: {action} on flag {flag_id}",
                    metadata={
                        "admin_id": admin_id,
                        "flag_id": flag_id,
                        "action": action,
                        "status": status
                    }
                )
                return True
            return False
        except Exception as e:
            print(f"Error moderating content: {e}")
            return False
    
    async def _update_flag_status(self, flag_id: str, status: str, admin_notes: Optional[str] = None) -> bool:
        """Update flag status in database"""
        try:
            if hasattr(self.admin_db.base_manager, 'db'):
                db = self.admin_db.base_manager.db
                if hasattr(db, 'cursor'):  # SQLite
                    cursor = db.cursor()
                    cursor.execute('''
                        UPDATE content_flags 
                        SET status = ?, admin_notes = ?
                        WHERE id = ?
                    ''', (status, admin_notes, flag_id))
                    db.commit()
                else:  # PostgreSQL
                    await db.execute('''
                        UPDATE content_flags 
                        SET status = $1, admin_notes = $2
                        WHERE id = $3
                    ''', (status, admin_notes, flag_id))
                return True
        except Exception as e:
            print(f"Error updating flag status: {e}")
            return False
    
    async def _store_moderation_action(self, action: ModerationAction) -> bool:
        """Store moderation action in database"""
        try:
            if hasattr(self.admin_db.base_manager, 'db'):
                db = self.admin_db.base_manager.db
                if hasattr(db, 'cursor'):  # SQLite
                    cursor = db.cursor()
                    cursor.execute('''
                        INSERT INTO moderation_actions 
                        (id, admin_id, admin_email, target_type, target_id, action_type, reason, status, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        action.id, action.admin_id, action.admin_email, action.target_type,
                        action.target_id, action.action_type, action.reason, action.status,
                        action.timestamp.isoformat()
                    ))
                    db.commit()
                else:  # PostgreSQL
                    await db.execute('''
                        INSERT INTO moderation_actions 
                        (id, admin_id, admin_email, target_type, target_id, action_type, reason, status, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ''', (
                        action.id, action.admin_id, action.admin_email, action.target_type,
                        action.target_id, action.action_type, action.reason, action.status,
                        action.timestamp
                    ))
                return True
        except Exception as e:
            print(f"Error storing moderation action: {e}")
            return False
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        try:
            analytics = await self.admin_db.get_analytics_data()
            
            # Calculate health score based on various metrics
            health_score = 100.0
            
            # Deduct points for high error rate
            if analytics.error_rate > 5.0:
                health_score -= (analytics.error_rate - 5.0) * 2
            
            # Deduct points for low activity
            if analytics.active_users_today < 10:
                health_score -= 10
            
            # Deduct points for high response time
            if analytics.average_response_time > 1.0:
                health_score -= (analytics.average_response_time - 1.0) * 20
            
            health_score = max(0.0, min(100.0, health_score))
            
            return {
                "health_score": round(health_score, 1),
                "status": "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical",
                "metrics": {
                    "error_rate": analytics.error_rate,
                    "response_time": analytics.average_response_time,
                    "uptime": analytics.system_uptime,
                    "active_users": analytics.active_users_today
                },
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error getting system health: {e}")
            return {
                "health_score": 0.0,
                "status": "unknown",
                "metrics": {},
                "last_updated": datetime.now().isoformat()
            } 