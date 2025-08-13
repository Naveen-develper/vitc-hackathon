import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Calendar, Clock, User, Phone, Mail, MapPin, Calendar as CalendarIcon } from 'lucide-react';
import DatePicker from 'react-datepicker';
import "react-datepicker/dist/react-datepicker.css";

const DoctorBooking = ({ doctor, onClose, onBookingSuccess }) => {
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [selectedTime, setSelectedTime] = useState('');
  const [patientName, setPatientName] = useState('');
  const [patientPhone, setPatientPhone] = useState('');
  const [patientEmail, setPatientEmail] = useState('');
  const [symptoms, setSymptoms] = useState('');
  const [loading, setLoading] = useState(false);
  const [availableSlots, setAvailableSlots] = useState([]);

  // Available time slots (9 AM to 6 PM, 30-minute intervals)
  const timeSlots = [
    '09:00', '09:30', '10:00', '10:30', '11:00', '11:30',
    '12:00', '12:30', '14:00', '14:30', '15:00', '15:30',
    '16:00', '16:30', '17:00', '17:30', '18:00'
  ];

  // Filter out past dates
  const filterPassedTime = (time) => {
    const currentDate = new Date();
    const selectedDateObj = new Date(selectedDate);
    
    if (selectedDateObj.toDateString() === currentDate.toDateString()) {
      const currentHour = currentDate.getHours();
      const currentMinute = currentDate.getMinutes();
      const [hour, minute] = time.split(':').map(Number);
      
      if (hour < currentHour || (hour === currentHour && minute <= currentMinute)) {
        return false;
      }
    }
    return true;
  };

  const handleDateChange = (date) => {
    setSelectedDate(date);
    setSelectedTime(''); // Reset time when date changes
  };

  const handleBooking = async () => {
    if (!selectedDate || !selectedTime || !patientName || !patientPhone || !patientEmail) {
      alert('Please fill in all required fields');
      return;
    }

    setLoading(true);
    try {
      // Create appointment data
      const appointmentData = {
        doctorId: doctor.id,
        doctorName: doctor.name,
        doctorEmail: doctor.email,
        patientName,
        patientPhone,
        patientEmail,
        appointmentDate: selectedDate.toISOString().split('T')[0],
        appointmentTime: selectedTime,
        symptoms,
        status: 'confirmed'
      };

      // Send appointment to backend
      const response = await fetch('http://127.0.0.1:8001/appointments/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(appointmentData),
      });

      if (!response.ok) {
        throw new Error('Failed to create appointment');
      }

      const savedAppointment = await response.json();
      console.log('Appointment saved:', savedAppointment);
      
      // Create Google Calendar event
      await createGoogleCalendarEvent(savedAppointment);
      
      onBookingSuccess(savedAppointment);
      onClose();
    } catch (error) {
      console.error('Booking failed:', error);
      alert('Booking failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const createGoogleCalendarEvent = async (appointmentData) => {
    try {
      // This would integrate with Google Calendar API
      // For now, we'll create a calendar link
      const startDateTime = new Date(selectedDate);
      const [hours, minutes] = selectedTime.split(':');
      startDateTime.setHours(parseInt(hours), parseInt(minutes), 0);
      
      const endDateTime = new Date(startDateTime);
      endDateTime.setMinutes(endDateTime.getMinutes() + 30); // 30-minute appointment
      
      const eventTitle = `Appointment with Dr. ${appointmentData.doctorName}`;
      const eventDescription = `
Patient: ${appointmentData.patientName}
Phone: ${appointmentData.patientPhone}
Email: ${appointmentData.patientEmail}
Symptoms: ${appointmentData.symptoms}
      `.trim();
      
      const calendarUrl = `https://calendar.google.com/calendar/render?action=TEMPLATE&text=${encodeURIComponent(eventTitle)}&dates=${startDateTime.toISOString().replace(/[-:]/g, '').split('.')[0]}Z/${endDateTime.toISOString().replace(/[-:]/g, '').split('.')[0]}Z&details=${encodeURIComponent(eventDescription)}&location=${encodeURIComponent(doctor.location)}`;
      
      // Open Google Calendar in new tab
      window.open(calendarUrl, '_blank');
      
    } catch (error) {
      console.error('Failed to create calendar event:', error);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 w-full max-w-lg max-h-[90vh] overflow-y-auto shadow-2xl animate-in zoom-in-95 duration-200">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Book Appointment
          </h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            ✕
          </Button>
        </div>

        {/* Doctor Info */}
        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg mb-6">
          <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
            Dr. {doctor.name}
          </h3>
          <div className="space-y-1 text-sm text-blue-800 dark:text-blue-200">
            <p className="flex items-center gap-2">
              <MapPin size={14} />
              {doctor.location}
            </p>
            <p className="flex items-center gap-2">
              <Phone size={14} />
              {doctor.phone}
            </p>
            <p className="flex items-center gap-2">
              <Mail size={14} />
              {doctor.email}
            </p>
          </div>
        </div>

        {/* Date and Time Selection */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <Label className="flex items-center gap-2 mb-2">
              <Calendar size={16} />
              Select Date
            </Label>
            <DatePicker
              selected={selectedDate}
              onChange={handleDateChange}
              minDate={new Date()}
              dateFormat="MMMM dd, yyyy"
              className="w-full p-2 border border-gray-300 rounded-md"
              placeholderText="Select appointment date"
            />
          </div>

          <div>
            <Label className="flex items-center gap-2 mb-2">
              <Clock size={16} />
              Select Time
            </Label>
            <div className="grid grid-cols-3 gap-2">
              {timeSlots.filter(filterPassedTime).map((time) => (
                <Button
                  key={time}
                  variant={selectedTime === time ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedTime(time)}
                  className="text-sm"
                >
                  {time}
                </Button>
              ))}
            </div>
          </div>
        </div>

        {/* Patient Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <Label className="flex items-center gap-2 mb-2">
              <User size={16} />
              Patient Name *
            </Label>
            <Input
              value={patientName}
              onChange={(e) => setPatientName(e.target.value)}
              placeholder="Enter patient name"
              required
            />
          </div>

          <div>
            <Label className="flex items-center gap-2 mb-2">
              <Phone size={16} />
              Phone Number *
            </Label>
            <Input
              value={patientPhone}
              onChange={(e) => setPatientPhone(e.target.value)}
              placeholder="Enter phone number"
              type="tel"
              required
            />
          </div>

          <div className="md:col-span-2">
            <Label className="flex items-center gap-2 mb-2">
              <Mail size={16} />
              Email *
            </Label>
            <Input
              value={patientEmail}
              onChange={(e) => setPatientEmail(e.target.value)}
              placeholder="Enter email address"
              type="email"
              required
            />
          </div>

          <div className="md:col-span-2">
            <Label className="mb-2">Symptoms/Reason for Visit</Label>
            <Textarea
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              placeholder="Describe symptoms or reason for appointment"
              rows={3}
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button
            variant="outline"
            onClick={onClose}
            className="flex-1 py-3"
          >
            Cancel
          </Button>
          <Button
            onClick={handleBooking}
            disabled={loading}
            className="flex-1 bg-blue-600 hover:bg-blue-700 py-3"
          >
            {loading ? (
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Booking...
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <CalendarIcon size={16} />
                Book Appointment
              </div>
            )}
          </Button>
        </div>

        <p className="text-xs text-gray-500 mt-4 text-center">
          Appointment will be added to your Google Calendar automatically
        </p>
      </div>
    </div>
  );
};

export default DoctorBooking; 