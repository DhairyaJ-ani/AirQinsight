'use client';

import React, { useMemo } from 'react';
import { Compass, BarChart2 } from 'lucide-react';
import DatePicker from './DatePicker';
import { getHazardCategory } from '@/lib/constants';

export interface RegionInfoProps {
  bbox: [number, number, number, number] | null;
  no2: Float32Array | null;
  availableDates?: string[];
  selectedDate?: string;
  onDateChange?: (date: string) => void;
  isLoadingDate?: boolean;
  pollutant?: string;
  pollutantName?: string;
  units?: string;
  measurementType?: string;
  datasetName?: string;
}

export default function RegionInfo({
  bbox,
  no2,
  availableDates = [],
  selectedDate = '',
  onDateChange,
  isLoadingDate = false,
  pollutant = 'no2',
  pollutantName = 'NO₂',
  units = 'µmol/m²',
  measurementType = 'satellite_column',
  datasetName,
}: RegionInfoProps) {
  // Determine region name based on bounding box
  const regionName = useMemo(() => {
    if (!bbox) return 'Global Coordinate Domain';
    const isMumbai =
      Math.abs(bbox[0] - 72.7) < 0.15 &&
      Math.abs(bbox[1] - 18.8) < 0.15 &&
      Math.abs(bbox[2] - 73.2) < 0.15 &&
      Math.abs(bbox[3] - 19.3) < 0.15;

    if (isMumbai) return 'Mumbai Metropolitan Region';
    return `Domain [${bbox[0].toFixed(2)}°, ${bbox[1].toFixed(2)}° to ${bbox[2].toFixed(2)}°, ${bbox[3].toFixed(2)}°]`;
  }, [bbox]);

  // Compute min, max, and mean statistics for active advected frame.
  // NoData (NaN) and infinite values are excluded; cloud-masked pixels
  // must not be treated as zero concentration.
  const stats = useMemo(() => {
    if (!no2 || no2.length === 0) {
      return { min: null, max: null, mean: null };
    }

    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let count = 0;

    for (let i = 0; i < no2.length; i++) {
      const val = no2[i];
      if (!isFinite(val) || isNaN(val)) continue; // skip NoData / cloud-masked
      if (val < min) min = val;
      if (val > max) max = val;
      sum += val;
      count++;
    }

    if (count === 0) return { min: null, max: null, mean: null };

    return {
      min: Math.round(min * 10) / 10,
      max: Math.round(max * 10) / 10,
      mean: Math.round((sum / count) * 10) / 10,
    };
  }, [no2]);

  // Determine hazard classification badge based on WHO/CPCB breakpoints
  const hazardBadge = useMemo(() => {
    const mean = stats.mean;
    if (mean === null) {
      return {
        label: 'Loading data…',
        textColor: 'text-zinc-400',
        bgColor: 'bg-zinc-800/60',
        borderColor: 'border-zinc-600/40',
        dotColor: 'bg-zinc-500',
      };
    }
    const cat = getHazardCategory(mean, pollutant);
    if (cat.category === 'NORMAL') {
      return {
        label: 'Good Air Quality',
        textColor: 'text-emerald-300',
        bgColor: 'bg-emerald-950/60',
        borderColor: 'border-emerald-500/40',
        dotColor: 'bg-emerald-400',
      };
    }
    if (cat.category === 'MODERATE') {
      return {
        label: 'Moderate Hazard',
        textColor: 'text-amber-300',
        bgColor: 'bg-amber-950/60',
        borderColor: 'border-amber-500/40',
        dotColor: 'bg-amber-400',
      };
    }
    if (cat.category === 'UNHEALTHY') {
      return {
        label: 'Unhealthy for Sensitive',
        textColor: 'text-orange-300',
        bgColor: 'bg-orange-950/60',
        borderColor: 'border-orange-500/40',
        dotColor: 'bg-orange-400',
      };
    }
    return {
      label: 'Critical / Hazardous',
      textColor: 'text-rose-300',
      bgColor: 'bg-rose-950/60',
      borderColor: 'border-rose-500/40',
      dotColor: 'bg-rose-500',
    };
  }, [stats, pollutant]);

  return (
    <div className="w-80 sm:w-96 bg-[#11141d]/95 backdrop-blur-md border border-[#242938] rounded-xl p-3.5 shadow-2xl text-[#f1f3f7] select-none">
      {/* Header with Source Pill */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2 pb-2 border-b border-[#242938] min-w-0">
        <div className="flex items-center gap-1.5 font-semibold text-xs tracking-wide text-white min-w-0 flex-1">
          <Compass className="w-3.5 h-3.5 text-blue-400 shrink-0" />
          <span className="truncate">{regionName}</span>
        </div>
        <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-blue-500/15 border border-blue-500/30 text-[10px] text-blue-300 font-mono shrink-0 whitespace-nowrap">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>ML Prediction</span>
        </div>
      </div>

      {/* Date Filling Form & Small Calendar Selector */}
      {availableDates.length > 0 && onDateChange && (
        <div className="mb-2.5 flex items-center justify-between gap-2 bg-[#161a26]/70 border border-[#242938] rounded-lg px-2.5 py-1.5">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-mono text-zinc-400 tracking-wider">
              Observation Date
            </span>
            <span className="text-[10px] text-zinc-500">
              {availableDates.length} days in test data
            </span>
          </div>
          <DatePicker
            availableDates={availableDates}
            selectedDate={selectedDate}
            onDateChange={onDateChange}
            isLoading={isLoadingDate}
          />
        </div>
      )}

      {/* Hazard Level Badge */}
      <div
        className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg border text-xs mb-3 font-medium ${hazardBadge.bgColor} ${hazardBadge.borderColor} ${hazardBadge.textColor}`}
      >
        <span className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${hazardBadge.dotColor}`} />
          {hazardBadge.label}
        </span>
        <span className="font-mono text-[11px] font-bold">
          {stats.mean !== null ? stats.mean : '–'} <span className="text-[9px] font-normal">{units}</span>
        </span>
      </div>

      {/* Telemetry Stats Grid */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="bg-[#161a26] border border-[#242938] rounded-lg p-1.5">
          <div className="text-[10px] text-zinc-400 uppercase font-mono">Min {pollutantName}</div>
          <div className="text-xs font-mono font-bold text-emerald-400 mt-0.5">
            {stats.min !== null ? stats.min : '–'}
          </div>
          <div className="text-[9px] text-zinc-400">{units}</div>
        </div>

        <div className="bg-[#161a26] border border-[#242938] rounded-lg p-1.5">
          <div className="text-[10px] text-zinc-400 uppercase font-mono flex items-center justify-center gap-0.5">
            <BarChart2 className="w-2.5 h-2.5 text-blue-400" />
            <span>Mean</span>
          </div>
          <div className="text-xs font-mono font-bold text-white mt-0.5">
            {stats.mean !== null ? stats.mean : '–'}
          </div>
          <div className="text-[9px] text-zinc-400">{units}</div>
        </div>

        <div className="bg-[#161a26] border border-[#242938] rounded-lg p-1.5">
          <div className="text-[10px] text-zinc-400 uppercase font-mono">Peak {pollutantName}</div>
          <div className="text-xs font-mono font-bold text-rose-400 mt-0.5">
            {stats.max !== null ? stats.max : '–'}
          </div>
          <div className="text-[9px] text-zinc-400">{units}</div>
        </div>
      </div>
    </div>
  );
}
