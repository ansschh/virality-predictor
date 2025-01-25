import React, { useState } from 'react';

function App() {
  const [formData, setFormData] = useState({
    text_length: "",
    title_length: "",
    language: "",
    post_hour: "",
    post_day: "",
    is_weekend: "",
    total_views: "",
    total_likes: "",
    total_comments: "",
    total_follows: "",
    total_bookmarks: "",
    unique_users: "",
    unique_countries: "",
    engagement_duration: ""
  });

  const [prediction, setPrediction] = useState(null);

  // Handle input changes
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch("https://virality-predictor.onrender.com/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text_length: Number(formData.text_length),
          title_length: Number(formData.title_length),
          language: Number(formData.language),
          post_hour: Number(formData.post_hour),
          post_day: Number(formData.post_day),
          is_weekend: Number(formData.is_weekend),
          total_views: Number(formData.total_views),
          total_likes: Number(formData.total_likes),
          total_comments: Number(formData.total_comments),
          total_follows: Number(formData.total_follows),
          total_bookmarks: Number(formData.total_bookmarks),
          unique_users: Number(formData.unique_users),
          unique_countries: Number(formData.unique_countries),
          engagement_duration: Number(formData.engagement_duration),
        }),
      });
      
      const data = await response.json();
      setPrediction(data.prediction);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  return (
    <div style={{ margin: "30px" }}>
      <h1>Virality Prediction</h1>
      <form onSubmit={handleSubmit} style={{ display: "grid", maxWidth: "400px", gap: "10px" }}>

        <label>Text Length:
          <input 
            type="number"
            name="text_length"
            value={formData.text_length}
            onChange={handleChange}
          />
        </label>

        <label>Title Length:
          <input 
            type="number"
            name="title_length"
            value={formData.title_length}
            onChange={handleChange}
          />
        </label>

        <label>Language (encoded):
          <input 
            type="number"
            name="language"
            value={formData.language}
            onChange={handleChange}
          />
        </label>

        <label>Post Hour (0-23):
          <input 
            type="number"
            name="post_hour"
            value={formData.post_hour}
            onChange={handleChange}
          />
        </label>

        <label>Post Day (0=Mon, 6=Sun):
          <input 
            type="number"
            name="post_day"
            value={formData.post_day}
            onChange={handleChange}
          />
        </label>

        <label>Is Weekend (0 or 1):
          <input 
            type="number"
            name="is_weekend"
            value={formData.is_weekend}
            onChange={handleChange}
          />
        </label>

        <label>Total Views:
          <input 
            type="number"
            name="total_views"
            value={formData.total_views}
            onChange={handleChange}
          />
        </label>

        <label>Total Likes:
          <input 
            type="number"
            name="total_likes"
            value={formData.total_likes}
            onChange={handleChange}
          />
        </label>

        <label>Total Comments:
          <input 
            type="number"
            name="total_comments"
            value={formData.total_comments}
            onChange={handleChange}
          />
        </label>

        <label>Total Follows:
          <input 
            type="number"
            name="total_follows"
            value={formData.total_follows}
            onChange={handleChange}
          />
        </label>

        <label>Total Bookmarks:
          <input 
            type="number"
            name="total_bookmarks"
            value={formData.total_bookmarks}
            onChange={handleChange}
          />
        </label>

        <label>Unique Users:
          <input 
            type="number"
            name="unique_users"
            value={formData.unique_users}
            onChange={handleChange}
          />
        </label>

        <label>Unique Countries:
          <input 
            type="number"
            name="unique_countries"
            value={formData.unique_countries}
            onChange={handleChange}
          />
        </label>

        <label>Engagement Duration (in seconds):
          <input 
            type="number"
            name="engagement_duration"
            value={formData.engagement_duration}
            onChange={handleChange}
          />
        </label>

        <button type="submit">Predict</button>
      </form>

      {prediction !== null && (
        <div style={{ marginTop: "20px" }}>
          <h2>Predicted Virality Score: {prediction.toFixed(2)}</h2>
        </div>
      )}
    </div>
  );
}

export default App;
