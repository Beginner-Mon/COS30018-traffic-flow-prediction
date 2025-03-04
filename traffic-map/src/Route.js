import React from "react";
const RouteComponent = ({ fetchData, reponse }) => {
    return (
        <div className="route-component">
            <h1>Route Component</h1>

            <p>select source</p>
            <select id="mySelect">
                <option value="">--Please choose an option--</option>
                <option value="option1">Option 1</option>
                <option value="option2">Option 2</option>
                <option value="option3">Option 3</option>
            </select>
            <p>select destination</p>
            <select id="mySelect">
                <option value="">--Please choose an option--</option>
                <option value="option1">Option 1</option>
                <option value="option2">Option 2</option>
                <option value="option3">Option 3</option>
            </select>
            <button onClick={fetchData}>get location</button>
            <div>
                {
                    reponse && (
                        <>
                            <p>latitude: {reponse.lat}</p>
                            <p>longitude: {reponse.long}</p>
                        </>
                    )

                }

            </div>
        </div>
    )
}
export default RouteComponent