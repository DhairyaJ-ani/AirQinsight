'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Database,
  Calendar,
  Satellite,
  Layers,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Info,
  ExternalLink,
  RefreshCw,
  FileCode,
  ShieldAlert,
} from 'lucide-react';
import { HistoricalDatasetInfo, PollutantAvailability } from '@/lib/types';

const API_BASE = 'http://localhost:8000';

export default function HistoricalDatasetSelector() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<HistoricalDatasetInfo[]>([]);
  const [pollutantsAvail, setPollutantsAvail] = useState<Record<string, PollutantAvailability>>({});
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('no2_61day_2025');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPollutantInfo, setSelectedPollutantInfo] = useState<PollutantAvailability | null>(null);

  const fetchHistoricalData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [dsRes, polRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/historical/datasets`),
        fetch(`${API_BASE}/api/v1/historical/pollutants/availability`),
      ]);

      if (dsRes.ok) {
        const dsData = await dsRes.json();
        setDatasets(dsData);
      } else {
        throw new Error(`Failed to load datasets: HTTP ${dsRes.status}`);
      }

      if (polRes.ok) {
        const polData = await polRes.json();
        setPollutantsAvail(polData);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Error fetching historical data';
      setError(msg);
      // Fallback local stub in case backend is warming up
      setDatasets([
        {
          id: 'no2_61day_2025',
          display_name: 'Sentinel-5P NO₂ — Mumbai (Nov–Dec 2025)',
          description:
            '61 daily Sentinel-5P TROPOMI tropospheric NO₂ column composites over Mumbai, 1 Nov–31 Dec 2025. Native 0.035° resolution (~3.9 km). Measurement type: satellite atmospheric column.',
          pollutant: 'no2',
          measurement_type: 'satellite_column',
          units: 'µmol/m²',
          provider: 'ESA Copernicus / Sentinel-5P TROPOMI OFFL L3',
          date_start: '2025-11-01',
          date_end: '2025-12-31',
          crs: 'EPSG:4326',
          native_resolution_deg: 0.035,
          native_resolution_km: 3.9,
          bounds: { west: 72.77, south: 18.865, east: 73.12, north: 19.32 },
          grid: { width: 10, height: 13 },
          nodata_convention: 'NaN (cloud-masked or missing)',
          purpose: 'main_historical',
        },
        {
          id: 'no2_samples_8day',
          display_name: 'Sentinel-5P NO₂ — Mumbai (8 weekly samples)',
          description:
            'Eight weekly Sentinel-5P TROPOMI tropospheric NO₂ column samples over Mumbai, 5 Nov–24 Dec 2025. Clear and partly cloudy conditions; max cloud gap ~35%. Purpose: upload demonstration and pipeline testing.',
          pollutant: 'no2',
          measurement_type: 'satellite_column',
          units: 'µmol/m²',
          provider: 'ESA Copernicus / Sentinel-5P TROPOMI OFFL L3',
          date_start: '2025-11-05',
          date_end: '2025-12-24',
          crs: 'EPSG:4326',
          native_resolution_deg: 0.035,
          native_resolution_km: 3.9,
          bounds: { west: 72.77, south: 18.865, east: 73.12, north: 19.32 },
          grid: { width: 10, height: 13 },
          nodata_convention: 'NaN (cloud-masked or missing)',
          purpose: 'test_samples',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistoricalData();
  }, []);

  const handleLaunchVisualization = (dsId: string, pol: string = 'no2') => {
    router.push(`/visualization?dataset=${encodeURIComponent(dsId)}&pollutant=${encodeURIComponent(pol)}`);
  };

  return (
    <div className="space-y-6">
      {/* Notice Banner explaining satellite column vs surface concentration */}
      <div className="rounded-xl border border-blue-500/30 bg-blue-500/10 p-4 text-blue-200">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-blue-400 mt-0.5 shrink-0" />
          <div className="space-y-1 text-xs">
            <span className="font-semibold text-blue-300">
              Multi-Pollutant Real Datasets Ready (Nov–Dec 2025)
            </span>
            <p className="text-zinc-300 leading-relaxed">
              All 6 pollutants have real 61-day historical datasets covering the Mumbai study domain.
              <strong className="text-amber-200"> NO₂</strong> is measured as tropospheric column density (µmol/m²) from Sentinel-5P TROPOMI.
              <strong className="text-emerald-200"> PM₂.₅, PM₁₀, SO₂, O₃, and CO</strong> are surface-level concentration analyses (µg/m³) from the Copernicus Atmosphere Monitoring Service (CAMS). Cloud-masked or missing observations are preserved as null.
            </p>
          </div>
        </div>
      </div>

      {/* Available Datasets Cards */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-blue-400" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-200">
              Bundled Multi-Pollutant Datasets (61 Daily Observations)
            </h2>
          </div>
          <button
            onClick={fetchHistoricalData}
            disabled={loading}
            className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {datasets.map((ds) => {
            const isSelected = selectedDatasetId === ds.id;
            return (
              <div
                key={ds.id}
                onClick={() => setSelectedDatasetId(ds.id)}
                className={`relative flex flex-col justify-between p-5 rounded-xl border transition-all cursor-pointer ${
                  isSelected
                    ? 'border-blue-500 bg-[#161b2b] shadow-lg shadow-blue-500/10'
                    : 'border-[#242938] bg-[#141721] hover:border-zinc-600'
                }`}
              >
                <div>
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
                        <Satellite className="w-4 h-4" />
                      </span>
                      <div>
                        <h3 className="text-sm font-bold text-zinc-100">{ds.display_name}</h3>
                        <p className="text-[11px] text-zinc-400 font-mono">ID: {ds.id}</p>
                      </div>
                    </div>
                    {isSelected && (
                      <span className="flex items-center gap-1 text-xs text-blue-400 font-medium bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Selected
                      </span>
                    )}
                  </div>

                  <p className="text-xs text-zinc-300 mt-3 leading-relaxed">
                    {ds.description}
                  </p>

                  {/* Metadata Specs Grid */}
                  <div className="grid grid-cols-2 gap-2 mt-4 text-[11px] font-mono bg-[#0d0f15]/70 p-3 rounded-lg border border-[#242938]">
                    <div>
                      <span className="text-zinc-500 block">Pollutant:</span>
                      <span className="text-zinc-200 uppercase font-semibold">{ds.pollutant}</span>
                    </div>
                    <div>
                      <span className="text-zinc-500 block">Measurement Type:</span>
                      <span className="text-cyan-300">{ds.measurement_type}</span>
                    </div>
                    <div>
                      <span className="text-zinc-500 block">Units:</span>
                      <span className="text-amber-300">{ds.units}</span>
                    </div>
                    <div>
                      <span className="text-zinc-500 block">Grid & Res:</span>
                      <span className="text-zinc-200">
                        {ds.grid.width}×{ds.grid.height} ({ds.native_resolution_km} km)
                      </span>
                    </div>
                    <div>
                      <span className="text-zinc-500 block">Date Range:</span>
                      <span className="text-zinc-200">
                        {ds.date_start} → {ds.date_end}
                      </span>
                    </div>
                    <div>
                      <span className="text-zinc-500 block">Purpose:</span>
                      <span className="text-zinc-300">{ds.purpose}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-5 pt-3 border-t border-[#242938] flex items-center justify-between gap-3">
                  <span className="text-[11px] text-zinc-400">
                    {ds.id === 'no2_61day_2025' ? '61 daily composites' : '8 weekly validation scenes'}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleLaunchVisualization(ds.id, ds.pollutant);
                    }}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md transition-all cursor-pointer group"
                  >
                    <span>Visualize Dataset</span>
                    <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Multi-Pollutant Readiness & Acquisition Status */}
      <div className="p-5 rounded-xl border border-[#242938] bg-[#141721] space-y-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            <div>
              <h3 className="text-sm font-semibold text-zinc-100">
                Multi-Pollutant Historical Coverage (Nov–Dec 2025 Window)
              </h3>
              <p className="text-xs text-zinc-400 mt-0.5">
                Scientific audit of satellite and ground observations across all 6 tracked pollutants.
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {[
            { key: 'no2', label: 'NO₂', status: 'available', desc: '61 TROPOMI rasters (µmol/m²)' },
            { key: 'pm25', label: 'PM₂.₅', status: 'available', desc: '61 CAMS rasters (µg/m³)' },
            { key: 'pm10', label: 'PM₁₀', status: 'available', desc: '61 CAMS rasters (µg/m³)' },
            { key: 'so2', label: 'SO₂', status: 'available', desc: '61 CAMS rasters (µg/m³)' },
            { key: 'co', label: 'CO', status: 'available', desc: '61 CAMS rasters (µg/m³)' },
            { key: 'o3', label: 'O₃', status: 'available', desc: '61 CAMS rasters (µg/m³)' },
          ].map((item) => {
            const availInfo = pollutantsAvail[item.key];
            const isAvail = availInfo?.has_raster_data_nov_dec_2025 ?? (item.status === 'available');

            return (
              <div
                key={item.key}
                onClick={() => setSelectedPollutantInfo(availInfo || null)}
                className={`p-3 rounded-lg border text-left cursor-pointer transition-all ${
                  isAvail
                    ? 'border-emerald-500/30 bg-emerald-500/5 hover:border-emerald-500/60'
                    : 'border-zinc-800 bg-[#0d0f15] hover:border-zinc-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-bold text-zinc-200">{item.label}</span>
                  <span
                    className={`w-2 h-2 rounded-full ${
                      isAvail ? 'bg-emerald-400 animate-pulse' : 'bg-amber-500'
                    }`}
                  />
                </div>
                <div className="mt-2">
                  <span
                    className={`inline-block text-[10px] uppercase font-mono px-1.5 py-0.5 rounded ${
                      isAvail
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : 'bg-zinc-800 text-zinc-400'
                    }`}
                  >
                    {isAvail ? 'Rasters Ready' : 'No 2025 Raster'}
                  </span>
                  <p className="text-[11px] text-zinc-400 mt-1 leading-tight line-clamp-2">
                    {item.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Selected Pollutant Reason Card or Helper */}
        {selectedPollutantInfo && !selectedPollutantInfo.has_raster_data_nov_dec_2025 && (
          <div className="p-4 rounded-lg bg-[#0d0f15] border border-zinc-800 text-xs text-zinc-300 space-y-2">
            <div className="flex items-center justify-between text-amber-400 font-semibold">
              <span className="flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4" />
                Data Availability Note for {selectedPollutantInfo.display_name} ({selectedPollutantInfo.formula})
              </span>
              <button
                onClick={() => setSelectedPollutantInfo(null)}
                className="text-zinc-500 hover:text-zinc-300"
              >
                ✕
              </button>
            </div>
            <p className="leading-relaxed">
              {selectedPollutantInfo.unavailability_reason ||
                'Satellite rasters for this pollutant during the Nov–Dec 2025 window are not bundled in this repo. Data cannot be fabricated.'}
            </p>
            {selectedPollutantInfo.acquisition_script && (
              <div className="flex items-center gap-2 pt-1 font-mono text-[11px] text-blue-400">
                <FileCode className="w-3.5 h-3.5" />
                <span>Acquisition workflow documented in: <code>{selectedPollutantInfo.acquisition_script}</code></span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
