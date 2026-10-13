import { getBrowserInfo } from "@visage/browser";

function App() {
  const info = getBrowserInfo();

  return (
    <main>
      <h1>Visage Playground</h1>
      <p>SDK Version: {info.version}</p>
      <p>
        Webcam Support: {info.webcamSupported ? "Available" : "Unavailable"}
      </p>
    </main>
  );
}

export default App;