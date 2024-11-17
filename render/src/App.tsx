import { Canvas } from '@react-three/fiber';
import { Scene } from './components/Scene';

function App() {
  return (
    <div className="fixed inset-0 overflow-hidden">
      {/* Canvas */}
      <Canvas className="w-full h-full bg-gradient-to-b from-gray-900 to-gray-800">
        <Scene />
      </Canvas>

      {/* Instructions */}
      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/50 to-transparent">
        <p className="text-center text-white text-sm">
          Click and drag to rotate • Scroll to zoom
        </p>
      </div>
    </div>
  );
}

export default App;