// src/MapComponent.js
import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import RouteComponent from './Route';

// Fix leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const MapComponent = () => {
    const position = [-37.86703, 145.09159]; // Center of Melbourne
    const [routes, setRoutes] = useState([]); // Store API routes
    const [source, setSource] = useState(""); // Default empty
    const [destination, setDestination] = useState(""); // Default empty
    const [scatData, setScatData] = useState({}); // Store SCAT details as a lookup object
    const [selectedRoute, setSelectedRoute] = useState(null); // Track selected route

    // Fetch all SCAT details on mount
    useEffect(() => {
        const fetchScatData = async () => {
            try {
                const response = await axios.get('http://localhost:5000/get_scats_number');
                const scatMap = response.data.scats.reduce((acc, scat) => {
                    acc[scat['SCATS Number']] = {
                        lat: scat.Latitude,
                        lng: scat.Longitude,
                        description: scat['Site Description'],
                        type: scat['Site Type'],
                        neighbours: scat.Neighbours
                    };
                    return acc;
                }, {});
                setScatData(scatMap);
            } catch (error) {
                console.error('Error fetching SCAT data:', error);
            }
        };
        fetchScatData();
    }, []);

    // Function to fetch routes
    const fetchRoutes = async () => {
        if (!source || !destination) {
            console.error('Source or destination not selected');
            setRoutes([]);
            return;
        }
        try {
            const response = await axios.post('http://localhost:5000/find_routes', {
                source: source,
                destination: destination,
            }, {
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (response.data.error) {
                console.error(response.data.error);
                setRoutes([]);
                setSelectedRoute(null); // Clear selected route on error
            } else {
                console.log('Routes response:', response.data);
                setRoutes(response.data.routes); // Store the routes
                setSelectedRoute(response.data.routes[0]); // Default to first route
            }
        } catch (error) {
            console.error('Error fetching routes:', error);
            setRoutes([]);
            setSelectedRoute(null);
        }
    };

    // Fetch routes on mount only if source and destination are set (optional)
    useEffect(() => {
        if (source && destination) {
            fetchRoutes();
        }
    }, [source, destination]);

    // Convert SCAT paths to lat/long positions for polylines
    const getPolylinePositions = (path) => {
        return path.map(scatId => {
            const scat = scatData[scatId];
            return scat ? [scat.lat, scat.lng] : position; // Fallback to center
        });
    };

    return (
        <>
            <MapContainer center={position} zoom={13} style={{ height: "100vh", width: "100%" }}>
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                {/* Markers for ALL SCAT points */}
                {Object.entries(scatData).map(([scatId, scat]) => (
                    <Marker
                        key={scatId}
                        position={[scat.lat, scat.lng]}
                    >
                        <Popup>
                            SCAT {scatId}<br />
                            Lat: {scat.lat.toFixed(5)}, Lng: {scat.lng.toFixed(5)}<br />
                            Description: {scat.description}<br />
                            Type: {scat.type}<br />
                            Neighbours: {scat.neighbours}
                        </Popup>
                    </Marker>
                ))}
                {/* Polyline for the selected route only */}
                {selectedRoute && (
                    <Polyline
                        positions={getPolylinePositions(selectedRoute.path)}
                        pathOptions={{ color: 'red', weight: 3 }}
                    />
                )}
            </MapContainer>
            <RouteComponent
                setSource={setSource}
                setDestination={setDestination}
                scatData={scatData}
                onFindRoutes={fetchRoutes}
                routes={routes}
                setSelectedRoute={setSelectedRoute} // Pass setter to select route
            />
        </>
    );
};

export default MapComponent;