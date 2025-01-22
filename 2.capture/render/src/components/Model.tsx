'use client';

import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { useLoader } from '@react-three/fiber';
import { useEffect } from 'react';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { ArrowHelper, Vector3 } from 'three';
import { useBonePositions } from '@/hooks/useBonePosition';

export function Model() {
  const gltf = useLoader(GLTFLoader, '/models/hand.glb');
  const bonesPositions = useBonePositions();

  useEffect(() => {
    if (!gltf) return;

    gltf.scene.traverse(node => {
      // Check if the current node's name exists in our bones positions
      if (bonesPositions[node.userData.name]) {
        const position = bonesPositions[node.userData.name];
        node.position.set(...position);
      }
    });
  }, [bonesPositions, gltf]);

  return (
    <>
      <PerspectiveCamera makeDefault position={[10, 6, 10]} />
      <OrbitControls makeDefault />

      <ambientLight intensity={0.5} />
      <directionalLight position={[0, 10, 10]} intensity={1.5} />

      {/* Axes helpers */}
      <primitive object={new ArrowHelper(new Vector3(1, 0, 0), new Vector3(0, 0, 0), 2, 0xff0000)} />
      <primitive object={new ArrowHelper(new Vector3(0, 1, 0), new Vector3(0, 0, 0), 2, 0x00ff00)} />
      <primitive object={new ArrowHelper(new Vector3(0, 0, 1), new Vector3(0, 0, 0), 2, 0x0000ff)} />

      <primitive object={gltf.scene} position={[0, 1, 0]} />
    </>
  );
}