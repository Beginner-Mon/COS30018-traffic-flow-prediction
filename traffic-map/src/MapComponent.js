// src/MapComponent.js
import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, CircleMarker } from 'react-leaflet';
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

// Custom red icon for destination
const destinationIcon = new L.Icon({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
    className: 'destination-marker' // For CSS tinting
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
                    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
                    attribution='Tiles © <a href="https://www.esri.com/">Esri</a> & contributors'
                />
                {/* Render SCAT points: Larger red CircleMarker for source, red Marker for destination, small CircleMarkers for others */}
                {Object.entries(scatData).map(([scatId, scat]) => (
                    scatId === source ? (
                        <CircleMarker
                            key={scatId}
                            center={[scat.lat, scat.lng]}
                            radius={8} 
                            pathOptions={{ color: 'red', fillColor: 'red', fillOpacity: 1 }}
                            zIndexOffset={1000} 
                        >
                            <Popup>
                                <span>SCAT {scatId}</span><br />
                                Lat: {scat.lat.toFixed(5)}, Lng: {scat.lng.toFixed(5)}<br />
                                Description: {scat.description}<br />
                                Type: {scat.type}<br />
                                Neighbours: {scat.neighbours}
                            </Popup>
                        </CircleMarker>
                    ) : scatId === destination ? (
                        <Marker
                            key={scatId}
                            position={[scat.lat, scat.lng]}
                            icon={destinationIcon}
                            zIndexOffset={1000} 
                        >
                            <Popup>
                                <span >SCAT {scatId}</span><br />
                                Lat: {scat.lat.toFixed(5)}, Lng: {scat.lng.toFixed(5)}<br />
                                Description: {scat.description}<br />
                                Type: {scat.type}<br />
                                Neighbours: {scat.neighbours}
                            </Popup>
                        </Marker>
                    ) : (
                        <CircleMarker
                            key={scatId}
                            center={[scat.lat, scat.lng]}
                            radius={4} // Smaller size for others
                            pathOptions={{ color: '#1E90FF', fillColor: 'white', fillOpacity: 1 }}
                        >
                            <Popup>
                                <span >SCAT {scatId}</span><br />
                                Lat: {scat.lat.toFixed(5)}, Lng: {scat.lng.toFixed(5)}<br />
                                Description: {scat.description}<br />
                                Type: {scat.type}<br />
                                Neighbours: {scat.neighbours}
                            </Popup>
                        </CircleMarker>
                    )
                ))}
                {/* Polyline for the selected route only */}
                {selectedRoute && (
                    <Polyline
                        positions={getPolylinePositions(selectedRoute.path)}
                        pathOptions={{ color: 'blue', weight: 4 }}
                        zIndexOffset={0} 
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