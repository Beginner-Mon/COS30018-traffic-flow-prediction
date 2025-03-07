import React from "react";
const RouteComponent = ({ fetchData, response }) => {

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
                    response && (
                        <>
                            <p>latitude: {response.data.lat}</p>
                            <p>longitude: {response.data.long}</p>
                        </>
                    )

                }

            </div>
        </div>
    )
}
export default RouteComponent