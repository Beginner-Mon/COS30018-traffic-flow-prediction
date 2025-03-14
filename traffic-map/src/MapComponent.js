// src/MapComponent.js
import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios'; // Import axios
import RouteComponent from './Route';

// Fix leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Mock SCAT data (replace with your actual data)
const scatData = {
    "970": [-37.86703, 145.09159],  // Example: Melbourne CBD
    "2000": [-37.8516827, 145.0943457],    // Slightly north-east
    "3685": [-37.85467, 145.09384],    // Slightly south-east
    // Add more SCAT IDs and their lat/long from traffic_network2.csv
};

const RoutingMachine = ({ waypoints }) => {
  const map = useMap();
  const routingRef = useRef();
  const isMounted = useRef(true);

  useEffect(() => {
    return () => {
      isMounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!map || waypoints.length < 2) return;

    const router = L.Routing.osrmv1({
      serviceUrl: 'https://router.project-osrm.org/route/v1'
    });

    const cleanupPrevious = () => {
      if (routingRef.current) {
        try {
          map.removeControl(routingRef.current);
          routingRef.current.getPlan().setWaypoints([]);
        } catch (error) {
          console.warn('Cleanup error:', error);
        }
        routingRef.current = null;
      }
    };

    cleanupPrevious();

    // Split waypoints into chunks of 3 (OSRM demo limit)
    const waypointChunks = [];
    for (let i = 0; i < waypoints.length; i += 2) {
      const chunk = waypoints.slice(i, i + 3);
      if (chunk.length > 1) waypointChunks.push(chunk);
    }

    const routingControls = waypointChunks.map(chunk => {
      const control = L.Routing.control({
        router,
        waypoints: chunk.map(pos => L.latLng(pos[0], pos[1])),
        routeWhileDragging: false,
        show: false,
        addWaypoints: false,
        createMarker: () => null,
        lineOptions: {
          styles: [{ color: '#FF0000', opacity: 0.7, weight: 5 }],
          extendToWaypoints: true,
          missingRouteTolerance: 1
        }
      });
      control.addTo(map);
      return control;
    });

    routingRef.current = routingControls;

    return () => {
      if (isMounted.current) {
        routingControls.forEach(control => {
          try {
            map.removeControl(control);
            control.getPlan().setWaypoints([]);
          } catch (error) {
            console.warn('Cleanup error:', error);
          }
        });
      }
    };
  }, [map, waypoints]);

  return null;
};

const MapComponent = () => {
    const position = [-37.814, 144.96332]; // Center of Melbourne
    const [routes, setRoutes] = useState([]); // Store API routes
    const [source, setSource] = useState("970"); // Default source
    const [destination, setDestination] = useState("2000"); // Default destination

    // Fetch routes from API using axios when source or destination changes
    useEffect(() => {
        const fetchRoutes = async () => {
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
                } else {
                    console.log(response)
                    setRoutes(response.data.routes); // Store the routes
                }
            } catch (error) {
                console.error('Error fetching routes:', error);
                setRoutes([]);
            }
        };

        fetchRoutes();
    }, [source, destination]); // Re-fetch when source or destination changes

    // Convert SCAT paths to lat/long positions for polylines
    const getPolylinePositions = (path) => {
        return path.map(scatId => scatData[scatId] || position); // Fallback to center if SCAT not found
    };

    return (
        <>
            <MapContainer center={position} zoom={13} style={{ height: "100vh", width: "100%" }}>
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                {/* Markers for all unique SCAT points in routes */}
                {routes.length > 0 && [...new Set(routes.flatMap(route => route.path))].map((scatId, index) => (
                    scatData[scatId] && (
                        <Marker key={index} position={scatData[scatId]}>
                            <Popup>
                                SCAT {scatId}: {scatData[scatId][0].toFixed(5)}, {scatData[scatId][1].toFixed(5)}
                            </Popup>
                        </Marker>
                    )
                ))}
                {/* Polylines for each route */}
                {routes.map((route, index) => (
                    <Polyline
                        key={index}
                        positions={getPolylinePositions(route.path)}
                        pathOptions={{ color: index === 0 ? 'red' : 'blue', weight: 3 }} // Highlight first route
                    />
                ))}
            </MapContainer>
            {/* Pass setSource and setDestination to RouteComponent */}
            <RouteComponent setSource={setSource} setDestination={setDestination} />
        </>
    );
};

export default MapComponent;