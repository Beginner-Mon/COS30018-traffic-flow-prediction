// src/MapComponent.js
import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, Polyline } from 'react-leaflet';
import L from 'leaflet';
import RouteComponent from './Route';
import axios from 'axios'
// Fix the default marker icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Function to generate random positions
const getRandomPosition = (latRange, lngRange) => {
    const lat = Math.random() * (latRange[1] - latRange[0]) + latRange[0];
    const lng = Math.random() * (lngRange[1] - lngRange[0]) + lngRange[0];
    return [lat, lng];
};

const MapComponent = () => {
    const position = [-37.814, 144.96332]; // Center of Melbourne
    const [randomPositions, setRandomPositions] = useState([]);
    const [polylinePositions, setPolylinePositions] = useState([]);

    // Define latitude and longitude ranges for Melbourne
    const latRange = [-37.840, -37.780]; // Approximate latitude range for Melbourne
    const lngRange = [144.910, 145.020]; // Approximate longitude range for Melbourne
    const [response, setResponse] = useState(null)

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
        // Generate random positions
        const positions = Array.from({ length: 10 }, () => getRandomPosition(latRange, lngRange));
        setRandomPositions(positions);
        setPolylinePositions(positions); // Set the same positions for the polyline
    }, []);


    return (
        <>
            <MapContainer center={position} zoom={13} style={{ height: "100vh", width: "100%" }}>
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                {randomPositions.map((pos, index) => (
                    <Marker key={index} position={pos}>
                        <Popup>
                            Random Position {index + 1}: {pos[0].toFixed(5)}, {pos[1].toFixed(5)}
                        </Popup>
                    </Marker>
                ))}
                <Circle center={position} radius={200} pathOptions={{ color: 'blue', fillOpacity: 0.5 }} />

                <Polyline positions={polylinePositions} pathOptions={{ color: 'blue', weight: 3 }} />
            </MapContainer>

            <RouteComponent fetchData={fetchData} response={response} />

        </>
    );
};

export default MapComponent;