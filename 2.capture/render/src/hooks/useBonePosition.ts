'use client';

import { useState, useEffect } from 'react';

export type BonesPositions = {
  [boneName: string]: [number, number, number];
};

export function useBonePositions() {
  const [bonesPositions, setBonesPositions] = useState<BonesPositions>({});

  useEffect(() => {
    const fetchBonesPositions = async () => {
      try {
        const response = await fetch('/api/bones');
        const data = await response.json();
        setBonesPositions(data);
      } catch (error) {
        console.error('Error fetching bones positions:', error);
      }
    };

    // Poll every 100ms (10 times per second)
    const interval = setInterval(fetchBonesPositions, 100);

    return () => clearInterval(interval);
  }, []);

  return bonesPositions;
}