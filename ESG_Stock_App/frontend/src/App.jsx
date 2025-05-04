import { useState } from 'react';
import axios from 'axios';

function App() {
  const [input, setInput] = useState('');
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async () => {
    const prices = input.split(',').map(Number);
    const res = await axios.post('http://localhost:8000/api/predict/stock', { sequence: prices });
    setPrediction(res.data.predicted_price);
  };

  return (
    <div className="p-8 max-w-xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">📈 Stock Price Predictor</h1>
      <textarea
        className="w-full border p-2"
        rows="5"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter 42 past stock prices, comma separated"
      />
      <button className="mt-4 bg-blue-600 text-white px-4 py-2" onClick={handleSubmit}>
        Predict Next Price
      </button>
      {prediction && (
        <p className="mt-4 text-green-600 font-semibold">📊 Predicted Price: {prediction.toFixed(2)}</p>
      )}
    </div>
  );
}

export default App;
