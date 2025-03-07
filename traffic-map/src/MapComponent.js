import React, { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet-routing-machine';

// Fix leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const getRandomPosition = (latRange, lngRange) => {
  const lat = Math.random() * (latRange[1] - latRange[0]) + latRange[0];
  const lng = Math.random() * (lngRange[1] - lngRange[0]) + lngRange[0];
  return [lat, lng];
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
  const position = [-37.814, 144.96332];
  const [randomPositions, setRandomPositions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  const latRange = [-37.840, -37.780];
  const lngRange = [144.910, 145.020];

  useEffect(() => {
    const generatePositions = () => {
      try {
        // Generate 6 points instead of 10 (better for demo server)
        const positions = Array.from({ length: 6 }, () => 
          getRandomPosition(latRange, lngRange)
        );
        setRandomPositions(positions);
      } catch (error) {
        console.error('Position generation failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    generatePositions();
  }, []);

  if (isLoading) return <div>Loading map...</div>;

  return (
    <MapContainer 
      center={position} 
      zoom={13} 
      style={{ height: "100vh", width: "100%" }}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; OpenStreetMap contributors'
      />
      
      {randomPositions.length >= 2 && (
        <RoutingMachine waypoints={randomPositions} />
      )}

      {randomPositions.map((pos, index) => (
        <Marker key={index} position={pos}>
          <Popup>
            Point {index + 1}<br />
            {pos[0].toFixed(5)}, {pos[1].toFixed(5)}
          </Popup>
        </Marker>
      ))}

      <Circle 
        center={position} 
        radius={200} 
        pathOptions={{ color: 'blue', fillOpacity: 0.2 }}
      />
    </MapContainer>
  );
};

export default MapComponent;