// src/Route.js
import React from 'react';

const RouteComponent = ({ scatData, setSource, setDestination, onFindRoutes, routes, setSelectedRoute }) => {
    const handleSubmit = (e) => {
        e.preventDefault();
        const source = e.target.source.value;
        const destination = e.target.destination.value;
        setSource(source); // Update source in MapComponent
        setDestination(destination); // Update destination in MapComponent
        onFindRoutes(); // Trigger route fetch in MapComponent
    };

    const handleRouteClick = (route) => {
        setSelectedRoute(route); // Set the clicked route as the selected one
    };

    return (
        <div className="route-component" >
            <h1>Route Component</h1>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Select Source:</label>
                    <select name="source" defaultValue="">
                        <option value="" disabled>Select a SCAT number</option>
                        {Object.keys(scatData).map(scatId => (
                            <option key={scatId} value={scatId}>
                                {scatId} - {scatData[scatId].description}
                            </option>
                        ))}
                    </select>
                </div>
                <div>
                    <label>Select Destination:</label>
                    <select name="destination" defaultValue="">
                        <option value="" disabled>Select a SCAT number</option>
                        {Object.keys(scatData).map(scatId => (
                            <option key={scatId} value={scatId}>
                                {scatId} - {scatData[scatId].description}
                            </option>
                        ))}
                    </select>
                </div>
                <button type="submit">Find Routes</button>
            </form>

            {/* Display all routes as clickable divs */}
            {routes.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                    <h2>Routes Found:</h2>
                    {routes.map((route, index) => (
                        <div
                            key={index}
                            onClick={() => handleRouteClick(route)}
                            style={{
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