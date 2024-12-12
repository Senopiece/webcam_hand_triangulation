import { NextResponse } from 'next/server';

type BonesPositions = {
  [boneName: string]: [number, number, number];
};

let latestBonesPositions: BonesPositions = {};

export async function GET() {
  return NextResponse.json(latestBonesPositions);
}

export async function POST(request: Request) {
  const data = await request.json();
  
  // Validate the incoming data
  const isValidData = Object.entries(data).every(([_, position]) => 
    Array.isArray(position) && 
    position.length === 3 && 
    position.every(coord => typeof coord === 'number')
  );

  if (!isValidData) {
    return NextResponse.json(
      { error: 'Invalid format. Expected: { "BoneName": [x, y, z], ... }' }, 
      { status: 400 }
    );
  }

  latestBonesPositions = data;
  return NextResponse.json({ success: true });
}