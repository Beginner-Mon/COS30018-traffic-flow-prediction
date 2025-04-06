import React, { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, CircleMarker } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import RouteComponent from './Route';
<<<<<<< HEAD
import axios from 'axios'
// Fix the default marker icon issue
=======

// Fix leaflet marker icons
>>>>>>> nguyen
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const destinationIcon = new L.Icon({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
});

const MapComponent = () => {
<<<<<<< HEAD
    const position = [-37.814, 144.96332]; // Center of Melbourne
    const [randomPositions, setRandomPositions] = useState([]);
    const [polylinePositions, setPolylinePositions] = useState([]);

    // Define latitude and longitude ranges for Melbourne
    const latRange = [-37.840, -37.780]; // Approximate latitude range for Melbourne
    const lngRange = [144.910, 145.020]; // Approximate longitude range for Melbourne
    const [response, setResponse] = useState(null)
=======
    const [initialPosition] = useState([-37.82327, 145.04517]);
    const [routes, setRoutes] = useState([]);
    const [source, setSource] = useState("");
    const [destination, setDestination] = useState("");
    const [scatData, setScatData] = useState({});
    const [selectedRoute, setSelectedRoute] = useState(null);
    const [location, setLocation] = useState('melbourne');
    const [routeMethod, setRouteMethod] = useState('route_finding'); // New state for route method
    const mapRef = useRef(null);
>>>>>>> nguyen

    const fetchData = async (event) => {
        event.preventDefault();

        const data = {
            "lat": position[0],
            "long": position[1]
        }

        try {
            const res = await axios.post('http://127.0.0.1:5000/submit', data);
            setResponse(res.data)
            console.log(res.data)

        } catch (error) {
            console.log(error)
        }
    }
    useEffect(() => {
        const newPosition = location === 'melbourne'
            ? [-37.86703, 145.09159]
            : [10.7769, 106.7009];

        if (mapRef.current) {
            mapRef.current.setView(newPosition, 13);
        }
    }, [location]);

    useEffect(() => {
        const fetchScatData = async () => {
            try {
                const response = await axios.get(`http://localhost:5000/get_scats_number?location=${location}`);
                const scatMap = response.data.scats.reduce((acc, scat) => {
                    acc[scat['SCATS Number']] = {
                        lat: scat.Latitude + 0.0012,
                        lng: scat.Longitude + 0.00125,
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
    }, [location]);

    const fetchRoutes = async (method) => {
        if (!source || !destination) {
            console.error('Source or destination not selected');
            setRoutes([]);
            return;
        }

        const endpoint = method === 'route_finding' ? '/find_routes' : '/find_routes_stgnn';
        const payload = method === 'route_finding'
            ? { source, destination, location }
            : { source, destination, start_time: "26-10-2006 8:00" }; // Hardcoded for now, adjust as needed

        try {
            const response = await axios.post(`http://localhost:5000${endpoint}`, payload, {
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (response.data.error) {
                console.error(response.data.error);
                setRoutes([]);
                setSelectedRoute(null);
            } else {
                console.log('Routes response:', response.data);
                setRoutes(response.data.routes);
                setSelectedRoute(response.data.routes[0]);
            }
        } catch (error) {
            console.error(`Error fetching routes from ${endpoint}:`, error);
            setRoutes([]);
            setSelectedRoute(null);
        }
    };

    useEffect(() => {
        if (source && destination) {
            fetchRoutes(routeMethod);
        }
    }, [source, destination, routeMethod]);

    const getPolylinePositions = (path) => {
        return path.map(scatId => {
            const scat = scatData[scatId];
            return scat ? [scat.lat, scat.lng] : [0, 0];
        });
    };

    const handleScatClick = (lat, lng) => {
        if (mapRef.current) {
            mapRef.current.setView([lat, lng], 13);
        }
    };

    const handleSelectionChange = (scatId, event) => {
        const value = event.target.value;
        if (value === 'source') {
            setSource(scatId);
        } else if (value === 'destination') {
            setDestination(scatId);
        }
        event.target.value = '';
    };


    return (
        <>
            <MapContainer
                center={initialPosition}
                zoom={13}
                minZoom={13}
                maxZoom={14}
                style={{ height: "100vh", width: "100%" }}
                ref={mapRef}
            >
                <TileLayer
                    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
                    attribution='Tiles © <a href="https://www.esri.com/">Esri</a> & contributors'
                />
                {Object.entries(scatData).map(([scatId, scat]) => (
                    scatId === source ? (
                        <CircleMarker
                            key={scatId}
                            center={[scat.lat, scat.lng]}
                            radius={8}
                            pathOptions={{ color: 'red', fillColor: 'red', fillOpacity: 1 }}
                            zIndexOffset={1000}
                            eventHandlers={{
                                click: () => handleScatClick(scat.lat, scat.lng)
                            }}
                        >
                            <Popup>
                                <span>SCAT {scatId}</span><br />
                                Lat: {scat.lat.toFixed(5)}, Lng: {scat.lng.toFixed(5)}<br />
                                Description: {scat.description}<br />
                                Type: {scat.type}<br />
                                Neighbours: {scat.neighbours}<br />
                                <select onChange={(e) => handleSelectionChange(scatId, e)} defaultValue="">
                                    <option value="" disabled>Select action</option>
                                    <option value="source">Set as Source</option>
                                    <option value="destination">Set as Destination</option>
                                </select>
                            </Popup>
                        </CircleMarker>
                    ) : scatId === destination ? (
                        <Marker
                            key={scatId}
                            position={[scat.lat, scat.lng]}
                            icon={destinationIcon}
                            zIndexOffset={1000}
                            eventHandlers={{
                                click: () => handleScatClick(scat.lat, scat.lng)
                            }}
                        >
                            <Popup>
                                <span>SCAT {scatId}</span><br />
                                Lat: {scat.lat.toFixed(5)}, Lng: {scat.lng.toFixed(5)}<br />
                                Description: {scat.description}<br />
                                Type: {scat.type}<br />
                                Neighbours: {scat.neighbours}<br />
                                <select onChange={(e) => handleSelectionChange(scatId, e)} defaultValue="">
                                    <option value="" disabled>Select action</option>
                                    <option value="source">Set as Source</option>
                                    <option value="destination">Set as Destination</option>
                                </select>
                            </Popup>
                        </Marker>
                    ) : (
                        <CircleMarker
                            key={scatId}
                            center={[scat.lat, scat.lng]}
                            radius={4}
                            pathOptions={{ color: '#1E90FF', fillColor: 'white', fillOpacity: 1 }}
                            eventHandlers={{
                                click: () => handleScatClick(scat.lat, scat.lng)
                            }}
                        >
                            <Popup>
                                <span>SCAT {scatId}</span><br />
                                Lat: {scat.lat.toFixed(5)}, Lng: {scat.lng.toFixed(5)}<br />
                                Description: {scat.description}<br />
                                Type: {scat.type}<br />
                                Neighbours: {scat.neighbours}<br />
                                <select onChange={(e) => handleSelectionChange(scatId, e)} defaultValue="">
                                    <option value="" disabled>Select action</option>
                                    <option value="source">Set as Source</option>
                                    <option value="destination">Set as Destination</option>
                                </select>
                            </Popup>
                        </CircleMarker>
                    )
                ))}
<<<<<<< HEAD
                <Circle center={position} radius={200} pathOptions={{ color: 'blue', fillOpacity: 0.5 }} />

                <Polyline positions={polylinePositions} pathOptions={{ color: 'blue', weight: 3 }} />
            </MapContainer>

            <RouteComponent fetchData={fetchData} response={response} />

=======
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
                setSelectedRoute={setSelectedRoute}
                setLocation={setLocation}
                setRouteMethod={setRouteMethod} // Pass setRouteMethod to RouteComponent
            />
>>>>>>> nguyen
        </>
    );
};

export default MapComponent;