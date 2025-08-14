import sqlite3
import asyncio
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
from .base import DatabaseManager
from models.admin_models import (
    UserActivityLog, SystemLog, AnalyticsData, ModerationAction, 
    ContentFlag, AdminUser, LogFilter, AnalyticsFilter
)

class AdminDatabaseManager:
    """Admin-specific database operations"""
    
    def __init__(self, base_manager: DatabaseManager):
        self.base_manager = base_manager
        # Get the database connection from the base manager
        if hasattr(base_manager, 'connection'):
            self.db = base_manager.connection  # SQLite uses 'connection'
        elif hasattr(base_manager, 'db'):
            self.db = base_manager.db  # PostgreSQL uses 'db'
        elif hasattr(base_manager, '_db'):
            self.db = base_manager._db
        else:
            self.db = None
    
    async def create_admin_tables(self) -> bool:
        """Create admin-specific database tables"""
        try:
            if isinstance(self.base_manager, SQLiteManager):
                return await self._create_sqlite_admin_tables()
            else:
                return await self._create_postgres_admin_tables()
        except Exception as e:
            print(f"Error creating admin tables: {e}")
            return False
    
    async def _create_sqlite_admin_tables(self) -> bool:
        """Create admin tables in SQLite"""
        try:
            cursor = self.db.cursor()
            
            # User Activity Logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_activity_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    user_email TEXT,
                    activity_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT
                )
            ''')
            
            # System Logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    stack_trace TEXT,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # Admin Users
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS admin_users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    last_login TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Moderation Actions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS moderation_actions (
                    id TEXT PRIMARY KEY,
                    admin_id TEXT NOT NULL,
                    admin_email TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    reason TEXT,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # Content Flags
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS content_flags (
                    id TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    content_id TEXT NOT NULL,
                    reporter_id TEXT,
                    reporter_email TEXT,
                    reason TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    admin_notes TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # Analytics Cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics_cache (
                    id TEXT PRIMARY KEY,
                    cache_key TEXT UNIQUE NOT NULL,
                    data TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            self.db.commit()
            return True
            
        except Exception as e:
            print(f"Error creating SQLite admin tables: {e}")
            return False
    
    async def _create_postgres_admin_tables(self) -> bool:
        """Create admin tables in PostgreSQL"""
        try:
            # User Activity Logs
            await self.db.execute('''
                CREATE TABLE IF NOT EXISTS user_activity_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    user_email TEXT,
                    activity_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL,
                    session_id TEXT
                )
            ''')
            
            # System Logs
            await self.db.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    stack_trace TEXT,
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL
                )
            ''')
            
            # Admin Users
            await self.db.execute('''
                CREATE TABLE IF NOT EXISTS admin_users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    permissions JSONB NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    last_login TIMESTAMP,
                    created_at TIMESTAMP NOT NULL
                )
            ''')
            
            # Moderation Actions
            await self.db.execute('''
                CREATE TABLE IF NOT EXISTS moderation_actions (
                    id TEXT PRIMARY KEY,
                    admin_id TEXT NOT NULL,
                    admin_email TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    reason TEXT,
                    status TEXT NOT NULL,
                    metadata JSONB,
                    timestamp TIMESTAMP NOT NULL
                )
            ''')
            
            # Content Flags
            await self.db.execute('''
                CREATE TABLE IF NOT EXISTS content_flags (
                    id TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    content_id TEXT NOT NULL,
                    reporter_id TEXT,
                    reporter_email TEXT,
                    reason TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    admin_notes TEXT,
                    timestamp TIMESTAMP NOT NULL
                )
            ''')
            
            # Analytics Cache
            await self.db.execute('''
                CREATE TABLE IF NOT EXISTS analytics_cache (
                    id TEXT PRIMARY KEY,
                    cache_key TEXT UNIQUE NOT NULL,
                    data JSONB NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            ''')
            
            return True
            
        except Exception as e:
            print(f"Error creating PostgreSQL admin tables: {e}")
            return False
    
    async def log_user_activity(self, activity: UserActivityLog) -> bool:
        """Log user activity"""
        try:
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute('''
                    INSERT INTO user_activity_logs 
                    (id, user_id, user_email, activity_type, description, ip_address, user_agent, metadata, timestamp, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    activity.id, activity.user_id, activity.user_email, activity.activity_type,
                    activity.description, activity.ip_address, activity.user_agent,
                    json.dumps(activity.metadata) if activity.metadata else None,
                    activity.timestamp.isoformat(), activity.session_id
                ))
                self.db.commit()
            else:
                await self.db.execute('''
                    INSERT INTO user_activity_logs 
                    (id, user_id, user_email, activity_type, description, ip_address, user_agent, metadata, timestamp, session_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ''', (
                    activity.id, activity.user_id, activity.user_email, activity.activity_type,
                    activity.description, activity.ip_address, activity.user_agent,
                    json.dumps(activity.metadata) if activity.metadata else None,
                    activity.timestamp, activity.session_id
                ))
            return True
        except Exception as e:
            print(f"Error logging user activity: {e}")
            return False
    
    async def log_system_event(self, log: SystemLog) -> bool:
        """Log system event"""
        try:
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute('''
                    INSERT INTO system_logs 
                    (id, level, component, message, stack_trace, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    log.id, log.level, log.component, log.message, log.stack_trace,
                    json.dumps(log.metadata) if log.metadata else None,
                    log.timestamp.isoformat()
                ))
                self.db.commit()
            else:
                await self.db.execute('''
                    INSERT INTO system_logs 
                    (id, level, component, message, stack_trace, metadata, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                ''', (
                    log.id, log.level, log.component, log.message, log.stack_trace,
                    json.dumps(log.metadata) if log.metadata else None,
                    log.timestamp
                ))
            return True
        except Exception as e:
            print(f"Error logging system event: {e}")
            return False
    
    async def get_analytics_data(self, filter_params: AnalyticsFilter = None) -> AnalyticsData:
        """Get system analytics data"""
        try:
            today = datetime.now().date()
            start_date = filter_params.start_date if filter_params else today
            end_date = filter_params.end_date if filter_params else today
            
            # Get basic counts
            total_users = await self._get_count("patients")
            total_analyses = await self._get_count("medical_records")
            total_appointments = await self._get_count("appointments")
            total_patients = await self._get_count("patients")
            
            # Get today's counts
            active_users_today = await self._get_count_today("user_activity_logs", "user_id")
            analyses_today = await self._get_count_today("medical_records")
            appointments_today = await self._get_count_today("appointments")
            patients_today = await self._get_count_today("patients")
            
            # Get Gemini API calls
            gemini_calls = await self._get_count_by_activity("analysis_request")
            gemini_calls_today = await self._get_count_today_by_activity("analysis_request")
            
            # Calculate system metrics
            system_uptime = await self._calculate_uptime()
            avg_response_time = await self._calculate_avg_response_time()
            error_rate = await self._calculate_error_rate()
            
            return AnalyticsData(
                total_users=total_users,
                active_users_today=active_users_today,
                total_analyses=total_analyses,
                analyses_today=analyses_today,
                total_appointments=total_appointments,
                appointments_today=appointments_today,
                total_patients=total_patients,
                patients_today=patients_today,
                system_uptime=system_uptime,
                average_response_time=avg_response_time,
                error_rate=error_rate,
                gemini_api_calls=gemini_calls,
                gemini_api_calls_today=gemini_calls_today
            )
            
        except Exception as e:
            print(f"Error getting analytics data: {e}")
            return AnalyticsData(
                total_users=0, active_users_today=0, total_analyses=0, analyses_today=0,
                total_appointments=0, appointments_today=0, total_patients=0, patients_today=0,
                system_uptime=0.0, average_response_time=0.0, error_rate=0.0,
                gemini_api_calls=0, gemini_api_calls_today=0
            )
    
    async def _get_count(self, table: str) -> int:
        """Get total count from table"""
        try:
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                return cursor.fetchone()[0]
            else:
                result = await self.db.fetchval(f"SELECT COUNT(*) FROM {table}")
                return result or 0
        except Exception as e:
            print(f"Error getting count from {table}: {e}")
            return 0
    
    async def _get_count_today(self, table: str, date_column: str = "created_at") -> int:
        """Get count from today"""
        try:
            today = datetime.now().date()
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE DATE({date_column}) = ?", (today,))
                return cursor.fetchone()[0]
            else:
                result = await self.db.fetchval(
                    f"SELECT COUNT(*) FROM {table} WHERE DATE({date_column}) = $1", 
                    today
                )
                return result or 0
        except Exception as e:
            print(f"Error getting today's count from {table}: {e}")
            return 0
    
    async def _get_count_by_activity(self, activity_type: str) -> int:
        """Get count by activity type"""
        try:
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute("SELECT COUNT(*) FROM user_activity_logs WHERE activity_type = ?", (activity_type,))
                return cursor.fetchone()[0]
            else:
                result = await self.db.fetchval(
                    "SELECT COUNT(*) FROM user_activity_logs WHERE activity_type = $1", 
                    activity_type
                )
                return result or 0
        except Exception as e:
            print(f"Error getting count by activity {activity_type}: {e}")
            return 0
    
    async def _get_count_today_by_activity(self, activity_type: str) -> int:
        """Get today's count by activity type"""
        try:
            today = datetime.now().date()
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM user_activity_logs WHERE activity_type = ? AND DATE(timestamp) = ?",
                    (activity_type, today)
                )
                return cursor.fetchone()[0]
            else:
                result = await self.db.fetchval(
                    "SELECT COUNT(*) FROM user_activity_logs WHERE activity_type = $1 AND DATE(timestamp) = $2",
                    activity_type, today
                )
                return result or 0
        except Exception as e:
            print(f"Error getting today's count by activity {activity_type}: {e}")
            return 0
    
    async def _calculate_uptime(self) -> float:
        """Calculate system uptime in hours"""
        try:
            # For now, return a mock value. In production, this would track actual uptime
            return 168.0  # 1 week
        except Exception as e:
            print(f"Error calculating uptime: {e}")
            return 0.0
    
    async def _calculate_avg_response_time(self) -> float:
        """Calculate average API response time"""
        try:
            # For now, return a mock value. In production, this would track actual response times
            return 0.5  # 500ms
        except Exception as e:
            print(f"Error calculating average response time: {e}")
            return 0.0
    
    async def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage"""
        try:
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute("SELECT COUNT(*) FROM system_logs WHERE level = 'error'")
                error_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM system_logs")
                total_count = cursor.fetchone()[0]
            else:
                error_count = await self.db.fetchval("SELECT COUNT(*) FROM system_logs WHERE level = 'error'")
                total_count = await self.db.fetchval("SELECT COUNT(*) FROM system_logs")
            
            if total_count == 0:
                return 0.0
            return (error_count / total_count) * 100
        except Exception as e:
            print(f"Error calculating error rate: {e}")
            return 0.0
    
    async def get_recent_activities(self, limit: int = 10) -> List[UserActivityLog]:
        """Get recent user activities"""
        try:
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute('''
                    SELECT id, user_id, user_email, activity_type, description, ip_address, 
                           user_agent, metadata, timestamp, session_id
                    FROM user_activity_logs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()
            else:
                rows = await self.db.fetch('''
                    SELECT id, user_id, user_email, activity_type, description, ip_address, 
                           user_agent, metadata, timestamp, session_id
                    FROM user_activity_logs 
                    ORDER BY timestamp DESC 
                    LIMIT $1
                ''', limit)
            
            activities = []
            for row in rows:
                activities.append(UserActivityLog(
                    id=row[0], user_id=row[1], user_email=row[2], activity_type=row[3],
                    description=row[4], ip_address=row[5], user_agent=row[6],
                    metadata=json.loads(row[7]) if row[7] else None,
                    timestamp=datetime.fromisoformat(row[8]) if isinstance(row[8], str) else row[8],
                    session_id=row[9]
                ))
            return activities
        except Exception as e:
            print(f"Error getting recent activities: {e}")
            return []
    
    async def get_recent_logs(self, limit: int = 10) -> List[SystemLog]:
        """Get recent system logs"""
        try:
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute('''
                    SELECT id, level, component, message, stack_trace, metadata, timestamp
                    FROM system_logs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()
            else:
                rows = await self.db.fetch('''
                    SELECT id, level, component, message, stack_trace, metadata, timestamp
                    FROM system_logs 
                    ORDER BY timestamp DESC 
                    LIMIT $1
                ''', limit)
            
            logs = []
            for row in rows:
                logs.append(SystemLog(
                    id=row[0], level=row[1], component=row[2], message=row[3],
                    stack_trace=row[4], metadata=json.loads(row[5]) if row[5] else None,
                    timestamp=datetime.fromisoformat(row[6]) if isinstance(row[6], str) else row[6]
                ))
            return logs
        except Exception as e:
            print(f"Error getting recent logs: {e}")
            return []
    
    async def get_pending_flags(self, limit: int = 10) -> List[ContentFlag]:
        """Get pending content flags"""
        try:
            if isinstance(self.base_manager, SQLiteManager):
                cursor = self.db.cursor()
                cursor.execute('''
                    SELECT id, content_type, content_id, reporter_id, reporter_email, 
                           reason, description, status, admin_notes, timestamp
                    FROM content_flags 
                    WHERE status = 'pending'
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                rows = cursor.fetchall()
            else:
                rows = await self.db.fetch('''
                    SELECT id, content_type, content_id, reporter_id, reporter_email, 
                           reason, description, status, admin_notes, timestamp
                    FROM content_flags 
                    WHERE status = 'pending'
                    ORDER BY timestamp DESC 
                    LIMIT $1
                ''', limit)
            
            flags = []
            for row in rows:
                flags.append(ContentFlag(
                    id=row[0], content_type=row[1], content_id=row[2], reporter_id=row[3],
                    reporter_email=row[4], reason=row[5], description=row[6], status=row[7],
                    admin_notes=row[8], timestamp=datetime.fromisoformat(row[9]) if isinstance(row[9], str) else row[9]
                ))
            return flags
        except Exception as e:
            print(f"Error getting pending flags: {e}")
            return []

# Import the SQLiteManager for type checking
from .sqlite_manager import SQLiteManager 