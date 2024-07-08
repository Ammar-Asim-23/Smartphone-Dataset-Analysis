import React, { useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";

const initialData = {
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
  model_choice: "best_rf_model" // default model choice
};

const PricePredictionForm = () => {
  const [formData, setFormData] = useState(initialData);

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
      toast.success(`Predicted Price: ${Math.round(predictedPrice)} Rs`);
      setFormData(initialData);
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
            key !== 'model_choice' ? (
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
            ) : null
          ))}
          <div className="form-group mb-3">
            <label htmlFor="model_choice" className="form-label">Model Choice</label>
            <select
              className="form-control"
              id="model_choice"
              name="model_choice"
              value={formData.model_choice}
              onChange={handleChange}
            >
              <option value="best_rf_model">Best RF Model</option>
              <option value="custom_linear_regression">Custom Linear Regression</option>
              <option value="custom_random_forest">Custom Random Forest</option>
              <option value="random_forest">Random Forest</option>
              <option value="linear_regression">Linear Regression</option>
            </select>
          </div>
          <button type="submit" className="btn btn-primary w-100">
            Predict Price
          </button>
        </form>
      </div>
    </div>
  );
};

export default PricePredictionForm;
