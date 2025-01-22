import dynamic from 'next/dynamic';
import { Suspense } from 'react';

// Dynamically import the ThreeScene component to avoid SSR issues
const ThreeScene = dynamic(() => import('./ThreeScene'), { ssr: false });

export default function Scene() {
  return (
    <div className="w-full h-full">
      <Suspense fallback={<div>Loading...</div>}>
        <ThreeScene />
      </Suspense>
    </div>
  );
}