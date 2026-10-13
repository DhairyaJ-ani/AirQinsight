'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { fromArrayBuffer } from 'geotiff';
import { DEFAULT_BBOX, generateSyntheticFrame } from '../visualization/_components/stubs';

export interface HomeGeoTiffData {
  /** Raw raster values (pollutant concentration or column density). */
  values: Float32Array;
  width: number;
  height: number;
  bbox: [number, number, number, number]; // [minLng, minLat, maxLng, maxLat]
  /** Which pollutant this data represents */
  pollutant: string;
}

/** @deprecated Use `data.values` instead — alias maintained for backward compatibility with NO2HeatmapOverlay */
export type LegacyHomeGeoTiffData = HomeGeoTiffData & { no2: Float32Array };

export interface UseHomeGeoTiffLoaderReturn {
  data: LegacyHomeGeoTiffData | null;
  isLoading: boolean;
  error: string | null;
  load: () => Promise<LegacyHomeGeoTiffData>;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

function authHeaders(): Record<string, string> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Loads a pollutant GeoTIFF grid for the home map.
 *
 * - For ``no2``: loads from ``/api/v1/downscale/geotiff?timestamp=<date>T00:00:00Z``
 * - For other pollutants: loads from ``/api/v1/downscale/geotiff?timestamp=<date>T00:00:00Z&pollutant=<key>``
 * - Falls back to a synthetic frame on HTTP error.
 *
 * The returned object includes both ``values`` (canonical) and ``no2`` (backward-compat alias
 * so existing ``NO2HeatmapOverlay`` calls that pass ``no2:`` continue to work).
 */
export function useHomeGeoTiffLoader(date?: string, pollutant = 'no2'): UseHomeGeoTiffLoaderReturn {
  const [data, setData] = useState<LegacyHomeGeoTiffData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cacheRef = useRef<Map<string, LegacyHomeGeoTiffData>>(new Map());
  const cacheKey = `${pollutant}__${date ?? 'latest'}`;

  const load = useCallback(async (): Promise<LegacyHomeGeoTiffData> => {
    const cached = cacheRef.current.get(cacheKey);
    if (cached) {
      setData(cached);
      return cached;
    }

    setIsLoading(true);
    setError(null);

    try {
      const pol = (pollutant || 'no2').toLowerCase();
      let url: string;
      if (date) {
        url = `${API_BASE}/api/v1/downscale/geotiff?timestamp=${encodeURIComponent(`${date}T00:00:00Z`)}`;
        if (pol !== 'no2') url += `&pollutant=${encodeURIComponent(pol)}`;
      } else {
        url = `${API_BASE}/api/v1/downscale/geotiff`;
        if (pol !== 'no2') url += `?pollutant=${encodeURIComponent(pol)}`;
      }

      const response = await fetch(url, { headers: authHeaders() });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      const tiff = await fromArrayBuffer(arrayBuffer);
      const image = await tiff.getImage();

      const rawBbox = image.getBoundingBox();
      const frameBbox: [number, number, number, number] = [rawBbox[0], rawBbox[1], rawBbox[2], rawBbox[3]];

      const width = image.getWidth();
      const height = image.getHeight();
      const rasters = await image.readRasters();

      if (rasters.length === 0) {
        throw new Error('GeoTIFF contained no raster bands');
      }

      const values = rasters[0] as Float32Array;

      const loadedData: LegacyHomeGeoTiffData = {
        values,
        // Backward-compat alias so callers using `no2:` continue to work
        get no2() { return values; },
        width,
        height,
        bbox: frameBbox,
        pollutant: pol,
      };

      cacheRef.current.set(cacheKey, loadedData);
      setData(loadedData);
      return loadedData;
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to fetch GeoTIFF';
      console.warn(`Home GeoTIFF loader falling back to synthetic frame: ${errorMsg}`);
      setError(`Backend error (${errorMsg}). Using synthetic baseline.`);

      // Fallback: smooth synthetic frame (100x100 grid) matching Visualization page
      const synth = generateSyntheticFrame(100, 100, DEFAULT_BBOX, 0);
      const values = synth.no2;
      const fallbackData: LegacyHomeGeoTiffData = {
        values,
        get no2() { return values; },
        width: synth.width,
        height: synth.height,
        bbox: synth.bbox,
        pollutant: (pollutant || 'no2').toLowerCase(),
      };

      cacheRef.current.set(cacheKey, fallbackData);
      setData(fallbackData);
      return fallbackData;
    } finally {
      setIsLoading(false);
    }
  }, [date, pollutant, cacheKey]);

  useEffect(() => {
    if (date === '') return; // uploaded dates are still loading
    load();
  }, [load, date, pollutant]);

  return { data, isLoading, error, load };
}
