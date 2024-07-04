import React, { useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";

const initalData = {
  resolution_height: "",
  processor_speed: "",
  screen_size: "",
  internal_memory: "",
  resolution_width: "",
  primary_camera_front: "",
  rating: "",
  ram_capacity: "",
  has_nfc: "",
  extended_memory_available: "",
  primary_camera_rear: "",
  os_ios: "",
  refresh_rate: "",
  has_5g: "",
  num_rear_cameras: "",
  num_cores: "",
  processor_brand_snapdragon: "",
}

const PricePredictionForm = () => {
  const [formData, setFormData] = useState(initalData);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(
        "http://localhost:5000/predict_price",
        formData
      );
      const predictedPrice = response.data.predicted_price;
      toast.success(`Predicted Price: ${Math.round(predictedPrice)}Rs`)
      setFormData(initalData)
    } catch (error) {
      toast.error("Error predicting price.");
      console.error("Error predicting price:", error);
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center mb-4">Smartphone Price Predictor</h1>
      <div className="card shadow p-4 mx-auto" style={{ maxWidth: "600px" }}>
        <form onSubmit={handleSubmit}>
          {Object.keys(formData).map((key) => (
            <div className="form-group mb-3" key={key}>
              <label htmlFor={key} className="form-label">
                {key.replace(/_/g, " ")}
              </label>
              <input
                type="text"
                className="form-control"
                id={key}
                name={key}
                value={formData[key]}
                onChange={handleChange}
                required
              />
            </div>
          ))}
          <button type="submit" className="btn btn-primary w-100">
            Predict Price
          </button>
        </form>
        
      </div>
    </div>
  );
};

export default PricePredictionForm;
