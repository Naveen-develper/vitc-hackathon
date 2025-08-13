import React, { useState, useEffect } from 'react';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Calendar, Clock, User, Phone, Mail, MapPin, Trash2, Edit, Search } from 'lucide-react';

const AppointmentManager = () => {
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchEmail, setSearchEmail] = useState('');
  const [filteredAppointments, setFilteredAppointments] = useState([]);

  useEffect(() => {
    if (searchEmail) {
      fetchAppointments(searchEmail);
    }
  }, [searchEmail]);

  const fetchAppointments = async (email) => {
    if (!email) return;
    
    setLoading(true);
    try {
      const response = await fetch(`http://127.0.0.1:8001/appointments/?patient_email=${email}`);
      if (response.ok) {
        const data = await response.json();
        setAppointments(data);
        setFilteredAppointments(data);
      } else {
        console.error('Failed to fetch appointments');
      }
    } catch (error) {
      console.error('Error fetching appointments:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    if (searchEmail) {
      fetchAppointments(searchEmail);
    }
  };

  const deleteAppointment = async (appointmentId) => {
    if (!confirm('Are you sure you want to cancel this appointment?')) return;
    
    try {
      const response = await fetch(`http://127.0.0.1:8001/appointments/${appointmentId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        setAppointments(appointments.filter(a => a.id !== appointmentId));
        setFilteredAppointments(filteredAppointments.filter(a => a.id !== appointmentId));
        alert('Appointment cancelled successfully');
      } else {
        alert('Failed to cancel appointment');
      }
    } catch (error) {
      console.error('Error cancelling appointment:', error);
      alert('Error cancelling appointment');
    }
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'confirmed':
        return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
      case 'cancelled':
        return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">
            Appointment Manager
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            View and manage your medical appointments
          </p>
        </div>

        {/* Search Section */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-6 mb-8 shadow-sm">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <Label htmlFor="email" className="block mb-2">
                Enter your email to view appointments
              </Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                <Input
                  id="email"
                  type="email"
                  placeholder="your.email@example.com"
                  value={searchEmail}
                  onChange={(e) => setSearchEmail(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <Button
              onClick={handleSearch}
              disabled={loading || !searchEmail}
              className="px-6 py-2"
            >
              {loading ? (
                <div className="flex items-center gap-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Loading...
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Search size={16} />
                  Search Appointments
                </div>
              )}
            </Button>
          </div>
        </div>

        {/* Appointments List */}
        {filteredAppointments.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              Your Appointments ({filteredAppointments.length})
            </h2>
            {filteredAppointments.map((appointment) => (
              <div
                key={appointment.id}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700"
              >
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
                          Dr. {appointment.doctor_name}
                        </h3>
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(appointment.status)}`}>
                            {appointment.status}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                          <Calendar size={16} />
                          <span className="font-medium">Date:</span>
                          <span>{formatDate(appointment.appointment_date)}</span>
                        </div>
                        <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                          <Clock size={16} />
                          <span className="font-medium">Time:</span>
                          <span>{appointment.appointment_time}</span>
                        </div>
                        <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                          <User size={16} />
                          <span className="font-medium">Patient:</span>
                          <span>{appointment.patient_name}</span>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                          <Phone size={16} />
                          <span className="font-medium">Phone:</span>
                          <span>{appointment.patient_phone}</span>
                        </div>
                        <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                          <Mail size={16} />
                          <span className="font-medium">Email:</span>
                          <span>{appointment.patient_email}</span>
                        </div>
                        {appointment.symptoms && (
                          <div className="text-gray-600 dark:text-gray-400">
                            <span className="font-medium">Symptoms:</span>
                            <p className="mt-1 text-sm">{appointment.symptoms}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-col gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2"
                      onClick={() => {
                        // Open Google Calendar with this appointment
                        const startDateTime = new Date(appointment.appointment_date);
                        const [hours, minutes] = appointment.appointment_time.split(':');
                        startDateTime.setHours(parseInt(hours), parseInt(minutes), 0);
                        
                        const endDateTime = new Date(startDateTime);
                        endDateTime.setMinutes(endDateTime.getMinutes() + 30);
                        
                        const calendarUrl = `https://calendar.google.com/calendar/render?action=TEMPLATE&text=Appointment with Dr. ${appointment.doctor_name}&dates=${startDateTime.toISOString().replace(/[-:]/g, '').split('.')[0]}Z/${endDateTime.toISOString().replace(/[-:]/g, '').split('.')[0]}Z&details=Patient: ${appointment.patient_name}%0APhone: ${appointment.patient_phone}%0AEmail: ${appointment.patient_email}%0ASymptoms: ${appointment.symptoms || 'Not specified'}`;
                        
                        window.open(calendarUrl, '_blank');
                      }}
                    >
                      <Calendar size={16} />
                      View in Calendar
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2 text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20"
                      onClick={() => deleteAppointment(appointment.id)}
                    >
                      <Trash2 size={16} />
                      Cancel
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && searchEmail && filteredAppointments.length === 0 && (
          <div className="text-center py-12">
            <Calendar className="mx-auto text-gray-400 mb-4" size={48} />
            <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
              No appointments found
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              No appointments found for {searchEmail}. Please check your email or book a new appointment.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AppointmentManager; 