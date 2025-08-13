import React, { useEffect, useState } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";
import {
  MapPin,
  UserSearch,
  Phone,
  Mail,
  Loader2,
  Map as MapIcon,
  Calendar,
} from "lucide-react";
import DoctorBooking from "./components/DoctorBooking";

const DoctorSearchPage = ({ defaultSpecialty = "" }) => {
  const [location, setLocation] = useState("");
  const [specialty, setSpecialty] = useState(defaultSpecialty);
  const [doctors, setDoctors] = useState([]);
  const [coords, setCoords] = useState(null);
  const [loading, setLoading] = useState(false);
  const [geoError, setGeoError] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [doctorsPerPage] = useState(6);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [showBooking, setShowBooking] = useState(false);
  const [searchAttempted, setSearchAttempted] = useState(false);

  const paginatedDoctors = doctors.slice(
    (currentPage - 1) * doctorsPerPage,
    currentPage * doctorsPerPage
  );

  // Try get current location on mount
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        setCoords({ lat: latitude, lng: longitude });

        try {
          const res = await axios.get(
            "https://nominatim.openstreetmap.org/reverse",
            {
              params: {
                lat: latitude,
                lon: longitude,
                format: "json",
              },
            }
          );

          const loc =
            res.data.address?.suburb ||
            res.data.address?.city ||
            res.data.address?.village ||
            "";
          setLocation(loc);
        } catch (err) {
          console.error("Error in reverse geocoding:", err);
        }
      },
      () => {
        console.warn("Geolocation permission denied or failed");
        setGeoError(true);
      }
    );
  }, []);

  // Geocode a location string to lat,lng
  const geocodeLocation = async (locStr) => {
    try {
      // Try multiple search strategies for better results
      const searchQueries = [
        locStr + ", India",
        locStr + ", Tamil Nadu, India",
        locStr,
        locStr + " city",
        locStr + " district"
      ];

      for (const query of searchQueries) {
        try {
          const res = await axios.get("https://nominatim.openstreetmap.org/search", {
            params: { 
              q: query, 
              format: "json", 
              limit: 1,
              addressdetails: 1,
              countrycodes: "in" // Focus on India
            },
          });
          
          if (res.data && res.data.length > 0) {
            const result = res.data[0];
            console.log("Geocoding successful:", result);
            return {
              lat: parseFloat(result.lat),
              lng: parseFloat(result.lon),
            };
          }
        } catch (e) {
          console.error(`Geocoding error for query "${query}":`, e);
          continue; // Try next query
        }
      }
      
      console.warn("All geocoding attempts failed for:", locStr);
      return null;
    } catch (e) {
      console.error("Geocoding error:", e);
      return null;
    }
  };

  const handleSearch = async () => {
    if (!location) {
      alert("Please enter a location.");
      return; 
    }
    setLoading(true);
    setDoctors([]);
    try {
      // Geocode input location to center the map
      let centerCoords = await geocodeLocation(location);
      
      // Fallback coordinates for common Indian cities
      if (!centerCoords) {
        const fallbackCoords = {
          'chennai': { lat: 13.0827, lng: 80.2707 },
          'mumbai': { lat: 19.0760, lng: 72.8777 },
          'delhi': { lat: 28.7041, lng: 77.1025 },
          'bangalore': { lat: 12.9716, lng: 77.5946 },
          'hyderabad': { lat: 17.3850, lng: 78.4867 },
          'kolkata': { lat: 22.5726, lng: 88.3639 },
          'pune': { lat: 18.5204, lng: 73.8567 },
          'ahmedabad': { lat: 23.0225, lng: 72.5714 }
        };
        
        const cityKey = location.toLowerCase().trim();
        if (fallbackCoords[cityKey]) {
          centerCoords = fallbackCoords[cityKey];
          console.log(`Using fallback coordinates for ${cityKey}:`, centerCoords);
        }
      }
      
      if (centerCoords) {
        setCoords(centerCoords);
        console.log("Map centered at:", centerCoords);
      } else {
        // Show user-friendly error instead of alert
        setGeoError(true);
        setCoords(null);
        return; // Don't proceed with doctor search
      }

      // Fetch doctors from backend
      const res = await axios.get("http://127.0.0.1:8001/api/search-doctors", {
        params: { location, specialty },
      });
      setDoctors(res.data);
      setSearchAttempted(true); // Mark that search was attempted
      console.log("Doctors found:", res.data);
    } catch (err) {
      console.error("Error fetching doctors:", err);
      // More user-friendly error message
      if (err.code === 'ECONNREFUSED') {
        alert("Cannot connect to server. Please make sure the backend is running.");
      } else {
        alert("Failed to fetch doctors. Please try again.");
      }
      setCoords(null);
      setSearchAttempted(true);
    } finally {
      setLoading(false);
    }
  };

  const handleBooking = (doctor) => {
    setSelectedDoctor(doctor);
    setShowBooking(true);
  };

  const handleBookingSuccess = (appointmentData) => {
    alert(`Appointment booked successfully with Dr. ${appointmentData.doctorName} on ${appointmentData.appointmentDate} at ${appointmentData.appointmentTime}!`);
    // You can add additional success handling here
  };

  const closeBooking = () => {
    setShowBooking(false);
    setSelectedDoctor(null);
  };

  return (
    <div className="p-6 space-y-6 bg-white dark:bg-black rounded-lg shadow-lg">
      <h1 className="text-3xl font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
        <UserSearch size={28} /> Find Doctors Near You
      </h1>

      <div className="flex flex-wrap gap-4 items-center">
        <div className="relative w-full sm:w-1/3">
          <MapPin className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400" size={20} />
          <Input
            placeholder="Enter city name (e.g., Chennai, Mumbai)"
            value={location}
            onChange={(e) => {
              setLocation(e.target.value);
              setGeoError(false); // Clear error when user types
            }}
            className="pl-10"
            disabled={loading}
          />
          <div className="absolute top-full left-0 right-0 mt-1 text-xs text-gray-500 dark:text-gray-400">
            Popular cities: Chennai, Mumbai, Delhi, Bangalore, Hyderabad
          </div>
        </div>

        <div className="relative w-full sm:w-1/3">
          <UserSearch className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400" size={20} />
          <Input
            placeholder="Specialty (e.g. Cardiologist)"
            value={specialty}
            onChange={(e) => setSpecialty(e.target.value)}
            className="pl-10"
            disabled={loading}
          />
        </div>

        <Button
          onClick={handleSearch}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-2 font-semibold"
        >
          {loading && <Loader2 className="animate-spin" size={18} />}
          Search
        </Button>
      </div>

      {geoError && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-medium text-red-800 dark:text-red-200">
                Location not found
              </h3>
              <div className="mt-2 text-sm text-red-700 dark:text-red-300">
                <p>We couldn't find "{location}" on the map. Try these suggestions:</p>
                <ul className="mt-2 list-disc list-inside space-y-1">
                  <li>Use the full city name (e.g., "Chennai" instead of "chennai")</li>
                  <li>Try nearby major cities like "Chennai", "Mumbai", "Delhi", "Bangalore"</li>
                  <li>Check spelling and use proper capitalization</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Map */}
      {coords ? (
        <MapContainer
          key={`${coords.lat}-${coords.lng}`} // remount on coords change
          center={[coords.lat, coords.lng]}
          zoom={13}
          className="h-[400px] w-full rounded-xl shadow-md"
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
          />
          {doctors.map((doc, idx) => {
            const lat = parseFloat(doc.lat);
            const lng = parseFloat(doc.lng);
            if (isNaN(lat) || isNaN(lng)) return null;
            return (
              <Marker key={idx} position={[lat, lng]}>
                <Popup>
                  <strong>{doc.name}</strong>
                  <br />
                  {doc.specialty}
                  <br />
                  {doc.phone}
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      ) : (
        <p className="text-gray-500 italic flex items-center gap-2">
          <MapIcon size={18} /> Map will show here after searching or location detection.
        </p>
      )}

      {/* List of doctors */}
      <div className="grid gap-6 pt-6">
              {!loading && !searchAttempted && (
        <div className="text-center py-12">
          <UserSearch className="mx-auto text-gray-400 mb-4" size={48} />
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            Ready to find doctors?
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Enter a city name above and click search to find doctors near you.
          </p>
        </div>
      )}
      
      {!loading && searchAttempted && doctors.length === 0 && (
        <div className="text-center py-12">
          <UserSearch className="mx-auto text-gray-400 mb-4" size={48} />
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No doctors found
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Try searching for a different location or specialty.
          </p>
        </div>
      )}
        {paginatedDoctors.map((doc, idx) => (
          <div
            key={idx}
            className="border border-gray-300 dark:border-gray-700 p-5 rounded-xl shadow-sm hover:shadow-lg transition-shadow duration-300"
          >
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-1 flex items-center gap-2">
              <UserSearch size={20} /> {doc.name}
            </h2>
            <p className="text-gray-600 dark:text-gray-300 mb-1">
              <strong>Specialty: </strong> {doc.specialty}
            </p>
            <p className="text-gray-600 dark:text-gray-300 mb-1 flex items-center gap-2">
              <MapPin size={16} /> {doc.location}
            </p>
            <p className="text-gray-600 dark:text-gray-300 mb-3 flex items-center gap-2">
              <Phone size={16} /> {doc.phone}
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                className="flex items-center gap-2 px-4 py-2"
                onClick={() => window.open(`mailto:${doc.email || ""}`, "_blank")}
              >
                <Mail size={16} /> Contact
              </Button>
              <Button
                onClick={() => handleBooking(doc)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white"
              >
                <Calendar size={16} /> Book Appointment
              </Button>
            </div>
          </div>
        ))}
        {doctors.length > doctorsPerPage && (
  <div className="flex justify-center gap-2 pt-4">
    <Button disabled={currentPage === 1} onClick={() => setCurrentPage(p => p - 1)}>Prev</Button>
    <Button disabled={currentPage * doctorsPerPage >= doctors.length} onClick={() => setCurrentPage(p => p + 1)}>Next</Button>
  </div>
        )}
      </div>

      {/* Booking Modal */}
      {showBooking && selectedDoctor && (
        <DoctorBooking
          doctor={selectedDoctor}
          onClose={closeBooking}
          onBookingSuccess={handleBookingSuccess}
        />
      )}
    </div>
  );
};

export default DoctorSearchPage;
