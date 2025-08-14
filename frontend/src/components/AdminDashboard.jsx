import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { 
  Activity, 
  Users, 
  FileText, 
  Calendar, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  TrendingUp,
  TrendingDown,
  Eye,
  Shield,
  BarChart3,
  Settings,
  Download
} from 'lucide-react';

const AdminDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [realtimeStats, setRealtimeStats] = useState(null);

  const BASE_API_URL = 'http://127.0.0.1:8001';

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchRealtimeStats, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${BASE_API_URL}/api/admin/dashboard`);
      if (!response.ok) throw new Error('Failed to fetch dashboard data');
      const data = await response.json();
      setDashboardData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchRealtimeStats = async () => {
    try {
      const response = await fetch(`${BASE_API_URL}/api/admin/stats/realtime`);
      if (response.ok) {
        const data = await response.json();
        setRealtimeStats(data);
      }
    } catch (err) {
      console.error('Error fetching realtime stats:', err);
    }
  };

  const getHealthStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getHealthStatusIcon = (status) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default: return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="w-96">
          <CardContent className="pt-6">
            <div className="text-center">
              <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Error Loading Dashboard</h3>
              <p className="text-gray-600 mb-4">{error}</p>
              <Button onClick={fetchDashboardData}>Retry</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const analytics = dashboardData?.analytics || {};
  const recentActivities = dashboardData?.recent_activities || [];
  const recentLogs = dashboardData?.recent_logs || [];
  const pendingFlags = dashboardData?.pending_flags || [];

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Admin Dashboard</h1>
          <p className="text-gray-600">Monitor system performance, user activities, and manage content</p>
        </div>

        {/* Health Status */}
        {realtimeStats && (
          <Card className="mb-6">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {getHealthStatusIcon(realtimeStats.analytics?.status || 'unknown')}
                  <div>
                    <h3 className="font-semibold">System Health</h3>
                    <p className={`text-sm ${getHealthStatusColor(realtimeStats.analytics?.status || 'unknown')}`}>
                      {realtimeStats.analytics?.status || 'Unknown'} Status
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-gray-900">
                    {realtimeStats.analytics?.health_score || 0}%
                  </p>
                  <p className="text-sm text-gray-600">Health Score</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center space-x-3">
                <Users className="h-8 w-8 text-blue-600" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Users</p>
                  <p className="text-2xl font-bold text-gray-900">{analytics.total_users || 0}</p>
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm">
                  <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
                  <span className="text-green-600">+{analytics.active_users_today || 0} today</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center space-x-3">
                <FileText className="h-8 w-8 text-green-600" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Total Analyses</p>
                  <p className="text-2xl font-bold text-gray-900">{analytics.total_analyses || 0}</p>
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm">
                  <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
                  <span className="text-green-600">+{analytics.analyses_today || 0} today</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center space-x-3">
                <Calendar className="h-8 w-8 text-purple-600" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Appointments</p>
                  <p className="text-2xl font-bold text-gray-900">{analytics.total_appointments || 0}</p>
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm">
                  <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
                  <span className="text-green-600">+{analytics.appointments_today || 0} today</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center space-x-3">
                <Activity className="h-8 w-8 text-orange-600" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Serious Cases</p>
                  <p className="text-2xl font-bold text-gray-900">{analytics.gemini_api_calls || 0}</p>
                </div>
              </div>
              <div className="mt-4">
                <div className="flex items-center text-sm">
                  <TrendingUp className="h-4 w-4 text-green-600 mr-1" />
                  <span className="text-green-600">+{analytics.gemini_api_calls_today || 0} today</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview" className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4" />
              <span>Overview</span>
            </TabsTrigger>
            <TabsTrigger value="logs" className="flex items-center space-x-2">
              <FileText className="h-4 w-4" />
              <span>Logs</span>
            </TabsTrigger>
            <TabsTrigger value="moderation" className="flex items-center space-x-2">
              <Shield className="h-4 w-4" />
              <span>Moderation</span>
            </TabsTrigger>
            <TabsTrigger value="settings" className="flex items-center space-x-2">
              <Settings className="h-4 w-4" />
              <span>Settings</span>
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Recent Activities */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Activity className="h-5 w-5" />
                    <span>Recent Activities</span>
                  </CardTitle>
                  <CardDescription>Latest user activities and system events</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {recentActivities.slice(0, 5).map((activity, index) => (
                      <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                        <div className="flex-shrink-0">
                          <Activity className="h-4 w-4 text-blue-600" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {activity.description}
                          </p>
                          <p className="text-xs text-gray-500">
                            {new Date(activity.timestamp).toLocaleString()}
                          </p>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          {activity.activity_type}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* System Performance */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <TrendingUp className="h-5 w-5" />
                    <span>System Performance</span>
                  </CardTitle>
                  <CardDescription>Key performance indicators</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Error Rate</span>
                        <span>{analytics.error_rate || 0}%</span>
                      </div>
                      <Progress value={analytics.error_rate || 0} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Response Time</span>
                        <span>{(analytics.average_response_time || 0).toFixed(2)}s</span>
                      </div>
                      <Progress value={(analytics.average_response_time || 0) * 100} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>System Uptime</span>
                        <span>{(analytics.system_uptime || 0).toFixed(1)}h</span>
                      </div>
                      <Progress value={Math.min((analytics.system_uptime || 0) / 24 * 100, 100)} className="h-2" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Logs Tab */}
          <TabsContent value="logs" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* User Activity Logs */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Users className="h-5 w-5" />
                    <span>User Activity Logs</span>
                  </CardTitle>
                  <CardDescription>Recent user activities and interactions</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {recentActivities.slice(0, 10).map((activity, index) => (
                      <div key={index} className="border-l-4 border-blue-500 pl-4 py-2">
                        <p className="text-sm font-medium">{activity.description}</p>
                        <p className="text-xs text-gray-500">
                          {activity.user_email || 'Anonymous'} • {new Date(activity.timestamp).toLocaleString()}
                        </p>
                        <Badge variant="outline" className="text-xs mt-1">
                          {activity.activity_type}
                        </Badge>
                      </div>
                    ))}
                  </div>
                  <Button variant="outline" className="w-full mt-4">
                    <Download className="h-4 w-4 mr-2" />
                    Export Logs
                  </Button>
                </CardContent>
              </Card>

              {/* System Logs */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Settings className="h-5 w-5" />
                    <span>System Logs</span>
                  </CardTitle>
                  <CardDescription>System events and error logs</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {recentLogs.slice(0, 10).map((log, index) => (
                      <div key={index} className={`border-l-4 pl-4 py-2 ${
                        log.level === 'error' ? 'border-red-500' :
                        log.level === 'warning' ? 'border-yellow-500' :
                        'border-green-500'
                      }`}>
                        <p className="text-sm font-medium">{log.message}</p>
                        <p className="text-xs text-gray-500">
                          {log.component} • {new Date(log.timestamp).toLocaleString()}
                        </p>
                        <Badge variant={log.level === 'error' ? 'destructive' : 'outline'} className="text-xs mt-1">
                          {log.level}
                        </Badge>
                      </div>
                    ))}
                  </div>
                  <Button variant="outline" className="w-full mt-4">
                    <Download className="h-4 w-4 mr-2" />
                    Export Logs
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Moderation Tab */}
          <TabsContent value="moderation" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Shield className="h-5 w-5" />
                  <span>Content Moderation</span>
                </CardTitle>
                <CardDescription>Review and manage flagged content</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {pendingFlags.length === 0 ? (
                    <div className="text-center py-8">
                      <Shield className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-600">No pending content flags</p>
                    </div>
                  ) : (
                    pendingFlags.map((flag, index) => (
                      <div key={index} className="border rounded-lg p-4 space-y-3">
                        <div className="flex justify-between items-start">
                          <div>
                            <h4 className="font-medium">{flag.content_type} Content</h4>
                            <p className="text-sm text-gray-600">ID: {flag.content_id}</p>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            {flag.status}
                          </Badge>
                        </div>
                        <div>
                          <p className="text-sm"><strong>Reason:</strong> {flag.reason}</p>
                          {flag.description && (
                            <p className="text-sm text-gray-600 mt-1">{flag.description}</p>
                          )}
                        </div>
                        <div className="flex justify-between items-center">
                          <p className="text-xs text-gray-500">
                            Reported by: {flag.reporter_email || 'Anonymous'}
                          </p>
                          <p className="text-xs text-gray-500">
                            {new Date(flag.timestamp).toLocaleString()}
                          </p>
                        </div>
                        <div className="flex space-x-2">
                          <Button size="sm" variant="outline" className="text-green-600">
                            Approve
                          </Button>
                          <Button size="sm" variant="outline" className="text-red-600">
                            Reject
                          </Button>
                          <Button size="sm" variant="outline">
                            Review
                          </Button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Settings className="h-5 w-5" />
                  <span>Admin Settings</span>
                </CardTitle>
                <CardDescription>Configure admin dashboard and system settings</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <div>
                    <h4 className="font-medium mb-3">System Configuration</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="text-sm font-medium">Log Retention (days)</label>
                        <input 
                          type="number" 
                          defaultValue="30" 
                          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium">Max Log Entries</label>
                        <input 
                          type="number" 
                          defaultValue="10000" 
                          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-3">Notifications</h4>
                    <div className="space-y-3">
                      <label className="flex items-center">
                        <input type="checkbox" defaultChecked className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                        <span className="ml-2 text-sm">Email notifications for errors</span>
                      </label>
                      <label className="flex items-center">
                        <input type="checkbox" defaultChecked className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                        <span className="ml-2 text-sm">Real-time dashboard updates</span>
                      </label>
                      <label className="flex items-center">
                        <input type="checkbox" className="rounded border-gray-300 text-blue-600 focus:ring-blue-500" />
                        <span className="ml-2 text-sm">Content flag alerts</span>
                      </label>
                    </div>
                  </div>
                  
                  <div className="flex space-x-3">
                    <Button>Save Settings</Button>
                    <Button variant="outline">Reset to Defaults</Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default AdminDashboard; 