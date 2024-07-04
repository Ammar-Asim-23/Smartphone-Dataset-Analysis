import './App.css';
import PricePredictionForm from './components/PricePredictionForm';
import Navbar from './components/Navbar';
import { ToastContainer,toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function App() {
  return (
    <div className="App">
    <Navbar />
    <div className="bg-light min-vh-100 d-flex align-items-center">
      <div className="container">
        <PricePredictionForm toast={toast} />
      </div>
    </div>
    <ToastContainer autoClose={1000} position='top-right' />
  </div>
  );
}

export default App;
