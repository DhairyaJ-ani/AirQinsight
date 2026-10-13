import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  const envKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '';
  return NextResponse.json({
    envKey,
    hasKey: envKey.trim().length > 0,
  });
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { key, persistToFile = false } = body;

    if (!key || typeof key !== 'string') {
      return NextResponse.json({ error: 'Valid API key is required' }, { status: 400 });
    }

    const trimmedKey = key.trim();

    // Strict validation to prevent environment file injection
    if (!/^[a-zA-Z0-9_-]{20,60}$/.test(trimmedKey)) {
      return NextResponse.json({ error: 'Invalid API key format' }, { status: 400 });
    }

    if (persistToFile) {
      if (process.env.NODE_ENV === 'production') {
        return NextResponse.json(
          { error: 'Environment file persistence is disabled in production environments' },
          { status: 403 }
        );
      }
      const frontendDir = process.cwd();
      const envLocalPath = path.join(frontendDir, '.env.local');
      const envPath = path.join(frontendDir, '.env');

      const updateEnvFile = (filePath: string) => {
        try {
          let content = '';
          if (fs.existsSync(filePath)) {
            content = fs.readFileSync(filePath, 'utf8');
          }
          if (content.includes('NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=')) {
            content = content.replace(
              /NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=.*/g,
              `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=${trimmedKey}`
            );
          } else {
            content += `\nNEXT_PUBLIC_GOOGLE_MAPS_API_KEY=${trimmedKey}\n`;
          }
          fs.writeFileSync(filePath, content, 'utf8');
        } catch (e) {
          console.warn(`Failed to update ${filePath}:`, e);
        }
      };

      updateEnvFile(envLocalPath);
      updateEnvFile(envPath);
    }

    return NextResponse.json({
      success: true,
      key: trimmedKey,
      persisted: persistToFile,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || 'Failed to update key' }, { status: 500 });
  }
}
