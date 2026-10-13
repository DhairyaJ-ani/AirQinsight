'use client';

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Map, setWorkerUrl } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import {
  Layers,
  ArrowLeft,
  Info,
  Database,
  AlertTriangle,
  FileCode,
  CheckCircle2,
  X,
  Satellite,
} from 'lucide-react';

import HeatmapCanvas from './HeatmapCanvas';
import PlayerControls from './PlayerControls';
import RegionInfo from './RegionInfo';
import ColorScale from './ColorScale';

import { useGeoTiffLoader } from './useGeoTiffLoader';
import { useAnimationEngine } from './useAnimationEngine';
import { DEFAULT_BBOX, USE_STUB_DATA, generateSyntheticTimestamps, DEFAULT_AVAILABLE_DATES } from './stubs';

// Initialize MapLibre Worker from local public bundle
setWorkerUrl('/maplibre/maplibre-gl-worker.mjs');

const API_BASE = 'http://localhost:8000';
const MUMBAI_CENTER: [number, number] = [72.85, 19.05];
const INITIAL_ZOOM = 9;

interface PollutantDef {
  key: string;
  name: string;
  formula: string;
  hasRasters: boolean;
  units: string;
  measurementType: string;
  sourceDesc: string;
}

const POLLUTANTS: PollutantDef[] = [
  {
    key: 'no2',
    name: 'Nitrogen Dioxide',
    formula: 'NO₂',
    hasRasters: true,
    units: 'µmol/m²',
    measurementType: 'satellite_column',
    sourceDesc: 'Sentinel-5P TROPOMI OFFL L3 Tropospheric Column',
  },
  {
    key: 'pm25',
    name: 'Fine Particulate Matter',
    formula: 'PM₂.₅',
    hasRasters: true,
    units: 'µg/m³',
    measurementType: 'surface_concentration',
    sourceDesc: 'CAMS European/Global NRT Analysis (Surface PM2.5)',
  },
  {
    key: 'pm10',
    name: 'Particulate Matter 10',
    formula: 'PM₁₀',
    hasRasters: true,
    units: 'µg/m³',
    measurementType: 'surface_concentration',
    sourceDesc: 'CAMS European/Global NRT Analysis (Surface PM10)',
  },
  {
    key: 'so2',
    name: 'Sulfur Dioxide',
    formula: 'SO₂',
    hasRasters: true,
    units: 'µg/m³',
    measurementType: 'surface_concentration',
    sourceDesc: 'CAMS European/Global NRT Analysis (Surface SO2)',
  },
  {
    key: 'co',
    name: 'Carbon Monoxide',
    formula: 'CO',
    hasRasters: true,
    units: 'µg/m³',
    measurementType: 'surface_concentration',
    sourceDesc: 'CAMS European/Global NRT Analysis (Surface CO)',
  },
  {
    key: 'o3',
    name: 'Ground-Level Ozone',
    formula: 'O₃',
    hasRasters: true,
    units: 'µg/m³',
    measurementType: 'surface_concentration',
    sourceDesc: 'CAMS European/Global NRT Analysis (Surface O3)',
  },
];

export default function VisualizationMap() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const initialDataset = searchParams?.get('dataset') || 'no2_61day_2025';
  const initialPollutant = searchParams?.get('pollutant') || 'no2';

  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const workerRef = useRef<Worker | null>(null);

  // Pollutant & Dataset selection state
  const [selectedPollutant, setSelectedPollutant] = useState<string>(initialPollutant);
  const [datasetId, setDatasetId] = useState<string>(initialDataset);
  const [showScienceTooltip, setShowScienceTooltip] = useState(false);
  const [unsupportedModalOpen, setUnsupportedModalOpen] = useState(false);
  const [unsupportedPol, setUnsupportedPol] = useState<PollutantDef | null>(null);

  // Application State
  const [availableDates, setAvailableDates] = useState<string[]>(DEFAULT_AVAILABLE_DATES);
  const [selectedDate, setSelectedDate] = useState<string>('2025-11-05');
  const [isLoadingDate, setIsLoadingDate] = useState(false);
  const [timestamps, setTimestamps] = useState<string[]>([]);
  const [currentNo2, setCurrentNo2] = useState<Float32Array | null>(null);
  const [currentBbox, setCurrentBbox] = useState<[number, number, number, number] | null>(DEFAULT_BBOX);
  const [gridWidth, setGridWidth] = useState(100);
  const [gridHeight, setGridHeight] = useState(100);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speedMultiplier, setSpeedMultiplier] = useState<1 | 2 | 4 | 10>(1);
  const [opacity, setOpacity] = useState(0.75);

  const activePolDef = useMemo(() => {
    return POLLUTANTS.find((p) => p.key === selectedPollutant) || POLLUTANTS[0];
  }, [selectedPollutant]);

  // 1. Initialize Web Worker on client mount
  useEffect(() => {
    try {
      const worker = new Worker(new URL('./NavierStokesWorker.ts', import.meta.url), {
        type: 'module',
      });
      workerRef.current = worker;
    } catch (err) {
      console.error('Failed to initialize NavierStokesWorker:', err);
    }

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, []);

  // 2. Fetch available dates for active dataset & pollutant
  useEffect(() => {
    let isCancelled = false;

    async function loadDates() {
      try {
        const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
        const headers: Record<string, string> = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        // Attempt historical coverage endpoint
        const coverageUrl = `${API_BASE}/api/v1/historical/${datasetId}/${selectedPollutant}/coverage`;
        const res = await fetch(coverageUrl, { headers });

        if (res.ok) {
          const data = await res.json();
          if (!isCancelled && Array.isArray(data?.dates)) {
            const validDates: string[] = data.dates
              .filter((d: { available: boolean }) => d.available)
              .map((d: { date: string }) => d.date);

            if (validDates.length > 0) {
              setAvailableDates(validDates);
              if (!validDates.includes(selectedDate)) {
                setSelectedDate(validDates[0]);
              }
              return;
            }
          }
        }

        // Fallback to downscale/dates
        const fallbackRes = await fetch(`${API_BASE}/api/v1/downscale/dates`, { headers });
        if (fallbackRes.ok) {
          const data = await fallbackRes.json();
          if (!isCancelled && Array.isArray(data?.dates) && data.dates.length > 0) {
            setAvailableDates(data.dates);
            if (!data.dates.includes(selectedDate)) {
              setSelectedDate(data.dates[0]);
            }
          }
        }
      } catch (err) {
        console.warn('Could not load dates from backend, using default date set:', err);
      }
    }

    loadDates();
    return () => {
      isCancelled = true;
    };
  }, [datasetId, selectedPollutant]);

  // 3. Update 30-minute timestamp intervals whenever selectedDate changes
  useEffect(() => {
    setIsLoadingDate(true);
    const newTimestamps = generateSyntheticTimestamps(48, selectedDate);
    setTimestamps(newTimestamps);
    const timer = setTimeout(() => {
      setIsLoadingDate(false);
    }, 200);
    return () => clearTimeout(timer);
  }, [selectedDate]);

  // 4. Initialize MapLibre GL Map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = new Map({
      container: mapContainerRef.current,
      style: 'https://tiles.openfreemap.org/styles/liberty',
      center: MUMBAI_CENTER,
      zoom: INITIAL_ZOOM,
      pitch: 0,
      bearing: 0,
      attributionControl: false,
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // 5. Data loader hook for GeoTIFF frame acquisition & caching
  const { frames, loadFrame } = useGeoTiffLoader(timestamps, datasetId, selectedPollutant);

  // 6. Callback triggered by animation loop whenever advected grid is ready for display
  const handleFrameReady = useCallback(
    (
      no2: Float32Array,
      bbox: [number, number, number, number],
      width: number,
      height: number
    ) => {
      setCurrentNo2(no2);
      setCurrentBbox(bbox);
      setGridWidth(width);
      setGridHeight(height);
    },
    []
  );

  // 7. Animation Engine hook driving timeline progress and Navier-Stokes worker iterations
  const { currentIndex, t, seek } = useAnimationEngine({
    timestamps,
    frames,
    loadFrame,
    workerRef,
    isPlaying,
    speedMultiplier,
    onFrameReady: handleFrameReady,
  });

  const handleSelectPollutant = (pol: PollutantDef) => {
    if (!pol.hasRasters) {
      setUnsupportedPol(pol);
      setUnsupportedModalOpen(true);
      return;
    }
    setSelectedPollutant(pol.key);
    if (pol.key === 'no2') {
      setDatasetId('no2_61day_2025');
    } else {
      setDatasetId(`${pol.key}_cams_61day_2025`);
    }
  };

  const handleSelectDataset = (newDsId: string) => {
    setDatasetId(newDsId);
  };

  return (
    <div className="relative w-full h-full overflow-hidden bg-[#0d0f15] select-none">
      {/* Layer 0: MapLibre GL Canvas Container */}
      <div ref={mapContainerRef} className="absolute inset-0 z-0 w-full h-full" />

      {/* Layer 10: Canvas 2D Advected Heatmap Overlay */}
      <HeatmapCanvas
        no2={currentNo2}
        bbox={currentBbox}
        width={gridWidth}
        height={gridHeight}
        mapRef={mapRef}
        opacity={opacity}
      />

      {/* Layer 20: Top Navigation & Multi-Pollutant Control Bar */}
      <div className="absolute top-4 left-4 right-4 z-30 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
        {/* Left: Back Button + Dataset Selector */}
        <div className="flex items-center gap-2 pointer-events-auto">
          <button
            onClick={() => router.back()}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-[#11141d]/95 hover:bg-[#1c2233] text-zinc-300 hover:text-white border border-[#242938] shadow-2xl backdrop-blur-md text-xs font-medium transition-colors cursor-pointer group"
            aria-label="Go back to previous page"
          >
            <ArrowLeft className="w-4 h-4 text-zinc-400 group-hover:text-blue-400 group-hover:-translate-x-0.5 transition-transform" />
            <span>Back</span>
          </button>

          {/* Dataset Switcher (NO₂ only) */}
          {selectedPollutant === 'no2' && (
            <div className="flex items-center bg-[#11141d]/95 border border-[#242938] rounded-xl p-1 shadow-2xl backdrop-blur-md">
              <button
                onClick={() => handleSelectDataset('no2_61day_2025')}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                  datasetId === 'no2_61day_2025'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <Database className="w-3.5 h-3.5" />
                <span>61-Day Full Series</span>
              </button>
              <button
                onClick={() => handleSelectDataset('no2_samples_8day')}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                  datasetId === 'no2_samples_8day'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <Satellite className="w-3.5 h-3.5" />
                <span>8-Day Samples</span>
              </button>
            </div>
          )}
        </div>

        {/* Center: Multi-Pollutant Selector Bar */}
        <div className="flex items-center gap-1 bg-[#11141d]/95 border border-[#242938] rounded-xl p-1 shadow-2xl backdrop-blur-md pointer-events-auto">
          {POLLUTANTS.map((pol) => {
            const isSelected = selectedPollutant === pol.key;
            return (
              <button
                key={pol.key}
                onClick={() => handleSelectPollutant(pol)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
                  isSelected
                    ? 'bg-blue-600 text-white shadow-md'
                    : pol.hasRasters
                    ? 'text-zinc-200 hover:bg-[#1a2030]'
                    : 'text-zinc-500 hover:text-zinc-300 hover:bg-[#161a26]'
                }`}
                title={pol.hasRasters ? `${pol.name} (Satellite Ready)` : `${pol.name} (Rasters Unavailable)`}
              >
                <span>{pol.formula}</span>
                {pol.hasRasters ? (
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                ) : (
                  <span className="w-1.5 h-1.5 rounded-full bg-zinc-600" />
                )}
              </button>
            );
          })}
        </div>

        {/* Right: Scientific Measurement Badge & Opacity */}
        <div className="flex items-center gap-2 pointer-events-auto">
          {/* Tropospheric Column Badge */}
          <div className="relative">
            <button
              onClick={() => setShowScienceTooltip((prev) => !prev)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#11141d]/95 border border-amber-500/30 text-amber-300 shadow-2xl backdrop-blur-md text-xs font-mono font-medium hover:bg-amber-500/10 transition-colors"
            >
              <Info className="w-3.5 h-3.5 text-amber-400" />
              <span className="hidden md:inline">Tropospheric Column</span>
              <span className="text-[10px] bg-amber-500/20 px-1 py-0.2 rounded border border-amber-500/30">
                {activePolDef.units}
              </span>
            </button>

            {/* Scientific Explanation Popover */}
            {showScienceTooltip && (
              <div className="absolute right-0 top-10 w-72 p-3.5 bg-[#141721] border border-amber-500/40 rounded-xl shadow-2xl text-xs text-zinc-300 space-y-2 z-50">
                <div className="flex items-center justify-between text-amber-300 font-semibold text-[11px]">
                  <span>Scientific Measurement Notice</span>
                  <button onClick={() => setShowScienceTooltip(false)} className="text-zinc-400 hover:text-zinc-200">
                    ✕
                  </button>
                </div>
                <p className="text-[11px] leading-relaxed">
                  Sentinel-5P TROPOMI measures <strong className="text-amber-200">atmospheric column density ({activePolDef.units})</strong> across the vertical troposphere.
                </p>
                <p className="text-[11px] text-zinc-400 leading-relaxed">
                  This represents integrated molecules per unit area, <span className="underline decoration-amber-400">not surface concentration (µg/m³)</span> at human breathing height. Cloud-masked pixels are masked to null.
                </p>
              </div>
            )}
          </div>

          {/* Heatmap Opacity Control */}
          <div className="bg-[#11141d]/95 backdrop-blur-md border border-[#242938] rounded-xl p-2.5 shadow-2xl text-[#f1f3f7] flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <input
              type="range"
              min="0"
              max="100"
              value={Math.round(opacity * 100)}
              onChange={(e) => setOpacity(Number(e.target.value) / 100)}
              className="w-20 sm:w-24 h-1.5 bg-[#1b202e] rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <span className="text-xs font-mono font-bold text-blue-400 w-8 text-right">
              {Math.round(opacity * 100)}%
            </span>
          </div>
        </div>
      </div>

      {/* Layer 20: Region & Telemetry Info Panel (Top-Left under Back button) */}
      <div className="absolute top-20 left-4 z-20">
        <RegionInfo
          bbox={currentBbox}
          no2={currentNo2}
          availableDates={availableDates}
          selectedDate={selectedDate}
          onDateChange={setSelectedDate}
          isLoadingDate={isLoadingDate}
          pollutant={activePolDef.key}
          pollutantName={activePolDef.formula}
          units={activePolDef.units}
          measurementType={activePolDef.measurementType}
          datasetName={datasetId === 'no2_61day_2025' ? '61-Day Sentinel-5P' : '8-Day Samples'}
        />
      </div>

      {/* Layer 20: NO₂ Atmospheric Legend Scale (Bottom-Right) */}
      <ColorScale
        pollutant={activePolDef.key}
        units={activePolDef.units}
        measurementType={activePolDef.measurementType}
      />

      {/* Layer 20: Interactive Player Controls Panel (Fixed Bottom) */}
      <PlayerControls
        timestamps={timestamps}
        currentIndex={currentIndex}
        t={t}
        isPlaying={isPlaying}
        speedMultiplier={speedMultiplier}
        onPlayPause={() => setIsPlaying((p) => !p)}
        onSpeedChange={setSpeedMultiplier}
        onSeek={seek}
        availableDates={availableDates}
        selectedDate={selectedDate}
        onDateChange={setSelectedDate}
        isLoadingDate={isLoadingDate}
      />

      {/* Unsupported Pollutant Modal */}
      {unsupportedModalOpen && unsupportedPol && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-md bg-[#141721] border border-amber-500/40 rounded-2xl p-6 shadow-2xl text-zinc-100 space-y-4">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <span className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400">
                  <AlertTriangle className="w-5 h-5" />
                </span>
                <div>
                  <h3 className="text-base font-bold text-zinc-100">
                    No Historical Rasters for {unsupportedPol.formula}
                  </h3>
                  <p className="text-xs text-zinc-400">{unsupportedPol.name}</p>
                </div>
              </div>
              <button
                onClick={() => setUnsupportedModalOpen(false)}
                className="text-zinc-400 hover:text-zinc-200 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs text-zinc-300 leading-relaxed bg-[#0d0f15] p-3.5 rounded-xl border border-[#242938]">
              <p>
                <strong className="text-amber-300">Scientific Integrity Notice:</strong> For the period 1 November–31 December 2025, this repository only contains validated <strong className="text-zinc-100">Sentinel-5P NO₂ tropospheric column rasters</strong> over Mumbai.
              </p>
              <p className="text-zinc-400">
                Rasters for {unsupportedPol.name} ({unsupportedPol.formula}) are not bundled in this repo and cannot be fabricated.
              </p>
              <div className="pt-2 border-t border-[#242938] space-y-1 font-mono text-[11px] text-blue-400">
                <div className="flex items-center gap-1.5">
                  <FileCode className="w-3.5 h-3.5" />
                  <span>Acquisition Workflow: <code>backend/scripts/acquire_historical_pollutants.py</code></span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={() => setUnsupportedModalOpen(false)}
                className="px-4 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-semibold transition-colors"
              >
                Close
              </button>
              <button
                onClick={() => {
                  setUnsupportedModalOpen(false);
                  setSelectedPollutant('no2');
                }}
                className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md transition-colors"
              >
                Switch to NO₂ (Available)
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
