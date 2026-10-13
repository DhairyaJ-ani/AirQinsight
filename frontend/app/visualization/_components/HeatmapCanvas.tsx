'use client';

import React, { useRef, useEffect, useCallback, useMemo } from 'react';
import type { Map as MapLibreMap } from 'maplibre-gl';
import chroma from 'chroma-js';
import { useMapSync } from './useMapSync';

export interface HeatmapCanvasProps {
  no2: Float32Array | null;
  bbox: [number, number, number, number] | null;
  width: number;
  height: number;
  mapRef: React.RefObject<MapLibreMap | null>;
  opacity: number;
  pollutant?: string;
}

interface PaletteConfig {
  colors: string[];
  domain: number[];
  max: number;
}

const POLLUTANT_PALETTES: Record<string, PaletteConfig> = {
  no2: {
    colors: ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023'],
    domain: [0, 40, 80, 120, 160, 200],
    max: 200,
  },
  pm25: {
    colors: ['#22d3ee', '#86efac', '#facc15', '#f97316', '#ef4444', '#7c3aed'],
    domain: [0, 15, 60, 100, 150, 250],
    max: 250,
  },
  pm10: {
    colors: ['#22d3ee', '#86efac', '#facc15', '#f97316', '#ef4444', '#7c3aed'],
    domain: [0, 25, 100, 180, 250, 400],
    max: 400,
  },
  so2: {
    colors: ['#a3e635', '#facc15', '#fb923c', '#ef4444', '#9f1239'],
    domain: [0, 40, 80, 200, 500],
    max: 500,
  },
  o3: {
    colors: ['#5eead4', '#a3e635', '#facc15', '#f97316', '#dc2626'],
    domain: [0, 60, 100, 160, 240],
    max: 240,
  },
  co: {
    colors: ['#67e8f9', '#86efac', '#fde047', '#fb923c', '#dc2626'],
    domain: [0, 500, 1000, 2000, 4000],
    max: 4000,
  },
};

function buildColorLUT(cfg: PaletteConfig): Uint8ClampedArray {
  const scale = chroma.scale(cfg.colors).domain(cfg.domain).mode('lch');
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const val = (i / 255.0) * cfg.max;
    const rgb = scale(val).rgb();
    const offset = i * 4;
    lut[offset] = rgb[0];
    lut[offset + 1] = rgb[1];
    lut[offset + 2] = rgb[2];
    lut[offset + 3] = 200; // ~78% base alpha
  }
  return lut;
}

export default function HeatmapCanvas({
  no2,
  bbox,
  width,
  height,
  mapRef,
  opacity,
  pollutant = 'no2',
}: HeatmapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Offscreen canvas for fast pixel rasterization
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  const polKey = (pollutant || 'no2').toLowerCase();
  const paletteCfg = POLLUTANT_PALETTES[polKey] || POLLUTANT_PALETTES.no2;
  const colorLUT = useMemo(() => buildColorLUT(paletteCfg), [paletteCfg]);

  // Main render pass drawing the rasterized grid projected onto the MapLibre screen bounds
  const renderHeatmap = useCallback(() => {
    const map = mapRef.current;
    if (!map || !bbox || !no2) return;

    // Initialize or resize offscreen buffer if grid dimensions change
    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement('canvas');
    }
    const offscreen = offscreenRef.current;
    if (offscreen.width !== width || offscreen.height !== height) {
      offscreen.width = width;
      offscreen.height = height;
    }

    const offscreenCtx = offscreen.getContext('2d');
    if (!offscreenCtx) return;

    // Populate offscreen ImageData from scalar values
    const imgData = offscreenCtx.createImageData(width, height);
    const data32 = new Uint32Array(imgData.data.buffer);
    const totalPixels = width * height;
    const maxVal = paletteCfg.max;

    for (let i = 0; i < totalPixels; i++) {
      const val = no2[i];
      if (isNaN(val) || val === null) {
        data32[i] = 0; // Transparent for NoData
        continue;
      }
      // Normalize to LUT index [0, 255]
      const lutIndex = Math.max(0, Math.min(255, Math.round((val / maxVal) * 255)));
      const offset = lutIndex * 4;

      const r = colorLUT[offset];
      const g = colorLUT[offset + 1];
      const b = colorLUT[offset + 2];
      const a = colorLUT[offset + 3];

      // Little-endian packed ABGR
      data32[i] = (a << 24) | (b << 16) | (g << 8) | r;
    }

    offscreenCtx.putImageData(imgData, 0, 0);

    // Project geographic corners into map CSS pixel coordinates
    // bbox is [minLng, minLat, maxLng, maxLat]
    const pTopLeft = map.project([bbox[0], bbox[3]]);
    const pBottomRight = map.project([bbox[2], bbox[1]]);

    const screenX = pTopLeft.x;
    const screenY = pTopLeft.y;
    const screenW = pBottomRight.x - pTopLeft.x;
    const screenH = pBottomRight.y - pTopLeft.y;

    const mainCanvas = canvasRef.current;
    if (!mainCanvas) return;

    const mainCtx = mainCanvas.getContext('2d');
    if (!mainCtx) return;

    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

    // Reset and clear main display canvas
    mainCtx.save();
    mainCtx.setTransform(1, 0, 0, 1, 0, 0);
    mainCtx.clearRect(0, 0, mainCanvas.width, mainCanvas.height);

    // Apply high-DPI scaling matrix
    mainCtx.scale(dpr, dpr);

    // Configure smooth bilinear resampling for natural plume gradient
    mainCtx.imageSmoothingEnabled = true;
    mainCtx.imageSmoothingQuality = 'high';
    mainCtx.globalAlpha = Math.max(0.0, Math.min(1.0, opacity));

    // Stretch and project grid onto map coordinates
    mainCtx.drawImage(offscreen, screenX, screenY, screenW, screenH);
    mainCtx.restore();
  }, [bbox, no2, width, height, mapRef, opacity, canvasRef, colorLUT, paletteCfg.max]);

  const { mapReady } = useMapSync(mapRef, canvasRef, renderHeatmap);

  // Trigger render pass whenever raster values, bounds, opacity, or map state updates
  useEffect(() => {
    if (mapReady) {
      renderHeatmap();
    }
  }, [mapReady, renderHeatmap]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none z-10 block"
      style={{
        width: '100%',
        height: '100%',
      }}
    />
  );
}
