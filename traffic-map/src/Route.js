// src/Route.js
import React, { useState } from 'react';

const RouteComponent = ({ scatData, setSource, setDestination, onFindRoutes, routes, setSelectedRoute, setLocation }) => {
    const [selectedLocation, setSelectedLocation] = useState('melbourne'); // Default to Melbourne

    const handleSubmit = (e) => {
        e.preventDefault();
        const source = e.target.source.value;
        const destination = e.target.destination.value;
        setSource(source);
        setDestination(destination);
        onFindRoutes();
    };

    const handleRouteClick = (route) => {
        setSelectedRoute(route);
    };

    const handleLocationChange = (e) => {
        const newLocation = e.target.value;
        setSelectedLocation(newLocation);
        setLocation(newLocation); // Pass the selected location to MapComponent
    };

    return (
        <div className="route-component">
            <h1 className='route-title'>

                <select className='select-title'
                    value={selectedLocation}
                    onChange={handleLocationChange}

                >
                    <option value="melbourne">Melbourne</option>
                    <option value="hcm">Ho Chi Minh City</option>
                </select>
            </h1>
            <form onSubmit={handleSubmit}>
                <div className='wrapper'>
                    <div className='route-container'>
                        <select className='route-select' name="source" defaultValue="">
                            <option value="" disabled>Select source</option>
                            {Object.keys(scatData).map(scatId => (
                                <option key={scatId} value={scatId}>
                                    {scatId} - {scatData[scatId].description}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className='route-container'>
                        <select className='route-select' name="destination" defaultValue="">
                            <option value="" disabled>Select destination</option>
                            {Object.keys(scatData).map(scatId => (
                                <option key={scatId} value={scatId}>
                                    {scatId} - {scatData[scatId].description}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>
                <button className='submit-button' type="submit">Find Routes</button>
            </form>

            {routes.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                    <h2>Routes Found:</h2>
                    {routes.map((route, index) => (
                        <div
                            key={index}
                            onClick={() => handleRouteClick(route)}
                            style={{
                                width: "420px",
                                padding: '10px',
                                marginBottom: '5px',
                                backgroundColor: '#f0f0f0',
                                cursor: 'pointer',
                                borderRadius: '5px',
                                transition: 'background-color 0.2s',
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#e0e0e0'}
                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f0f0f0'}
                        >
                            <strong>Route {route.route_number}:</strong> {route.path.join(' -> ')}<br />
                            <span>Travel Time: {route.travel_time_minutes.toFixed(2)} minutes</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default RouteComponent;