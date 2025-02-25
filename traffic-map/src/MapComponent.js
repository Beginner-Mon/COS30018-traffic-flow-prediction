// src/MapComponent.js
import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, CircleMarker, Polyline, Polygon } from 'react-leaflet';
import L from 'leaflet';

// Fix the default marker icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const MapComponent = () => {
    const position = [-37.8270, 145.0523]; // Example coordinates (latitude, longitude)
    const routes = [
        [-37.8270, 145.0523],
        [51.51, -0.1],
        [51.51, -0.12],
    ]
    return (
        <MapContainer center={position} zoom={12} style={{ height: "100vh", width: "100%" }}>
            <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            <Marker position={position} >
                <Popup>
                    A pretty CSS3 popup. <br /> Easily customizable.
                </Popup>
            </Marker>

            <Polyline positions={routes} color="blue" />

        </MapContainer>
    );
};

export default MapComponent;