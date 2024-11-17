import { Html, OrbitControls, PerspectiveCamera, useProgress } from '@react-three/drei';
import { useLoader } from '@react-three/fiber';
import { Suspense } from 'react';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { ArrowHelper, Vector3 } from 'three/src/Three.js';


function Loader() {
    const { progress } = useProgress()
    return <Html center>{progress} % loaded</Html>
}

export function Scene() {
    const gltf = useLoader(GLTFLoader, '/models/hand.glb')

    gltf.scene.traverse(n => {
        if (n.name === "Whirst") {
            console.log('found bone: ', n)
            n.rotation.set(0, 0, 0)
        }
    })

    return (
        <Suspense fallback={<Loader />}>
            <PerspectiveCamera makeDefault position={[10, 6, 10]} />
            <OrbitControls makeDefault />

            <ambientLight intensity={0.5} />
            <directionalLight position={[0, 10, 10]} intensity={1.5} />

            {/* Arrows for X, Y, Z axes */}
            <primitive
                object={new ArrowHelper(new Vector3(1, 0, 0), new Vector3(0, 0, 0), 2, 0xff0000)}
                position={[0, 0, 0]} // X arrow (red)
            />
            <primitive
                object={new ArrowHelper(new Vector3(0, 1, 0), new Vector3(0, 0, 0), 2, 0x00ff00)}
                position={[0, 0, 0]} // Y arrow (green)
            />
            <primitive
                object={new ArrowHelper(new Vector3(0, 0, 1), new Vector3(0, 0, 0), 2, 0x0000ff)}
                position={[0, 0, 0]} // Z arrow (blue)
            />

            <primitive
                object={gltf.scene}
                position={[0, 1, 0]}
            />
        </Suspense>
    );
};
