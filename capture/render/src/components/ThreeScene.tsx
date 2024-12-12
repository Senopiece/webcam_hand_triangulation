'use client';

import { Canvas } from '@react-three/fiber';
import { Model } from './Model';

export default function ThreeScene() {
  return (
    <Canvas>
      <Model />
    </Canvas>
  );
}