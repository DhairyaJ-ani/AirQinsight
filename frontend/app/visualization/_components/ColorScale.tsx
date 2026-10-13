'use client';

import React from 'react';

interface ColorStop {
  value: number;
  label: string;
  color: string;
}

const POLLUTANT_COLOR_STOPS: Record<string, { stops: ColorStop[]; gradient: string }> = {
  no2: {
    stops: [
      { value: 200, label: 'Hazardous', color: '#7e0023' },
      { value: 160, label: 'Severe', color: '#8f3f97' },
      { value: 120, label: 'Very Unhealthy', color: '#ff0000' },
      { value: 80, label: 'Unhealthy', color: '#ff7e00' },
      { value: 40, label: 'Moderate', color: '#ffff00' },
      { value: 0, label: 'Good', color: '#00e400' },
    ],
    gradient: 'linear-gradient(to bottom, #7e0023 0%, #8f3f97 20%, #ff0000 40%, #ff7e00 60%, #ffff00 80%, #00e400 100%)',
  },
  pm25: {
    stops: [
      { value: 250, label: 'Severe', color: '#7c3aed' },
      { value: 150, label: 'Hazardous', color: '#ef4444' },
      { value: 100, label: 'Poor', color: '#f97316' },
      { value: 60, label: 'Moderate', color: '#facc15' },
      { value: 15, label: 'Satisfactory', color: '#86efac' },
      { value: 0, label: 'Good', color: '#22d3ee' },
    ],
    gradient: 'linear-gradient(to bottom, #7c3aed 0%, #ef4444 20%, #f97316 40%, #facc15 60%, #86efac 80%, #22d3ee 100%)',
  },
  pm10: {
    stops: [
      { value: 400, label: 'Severe', color: '#7c3aed' },
      { value: 250, label: 'Hazardous', color: '#ef4444' },
      { value: 180, label: 'Poor', color: '#f97316' },
      { value: 100, label: 'Moderate', color: '#facc15' },
      { value: 25, label: 'Satisfactory', color: '#86efac' },
      { value: 0, label: 'Good', color: '#22d3ee' },
    ],
    gradient: 'linear-gradient(to bottom, #7c3aed 0%, #ef4444 20%, #f97316 40%, #facc15 60%, #86efac 80%, #22d3ee 100%)',
  },
  so2: {
    stops: [
      { value: 500, label: 'Hazardous', color: '#9f1239' },
      { value: 200, label: 'Very Poor', color: '#ef4444' },
      { value: 80, label: 'Moderate', color: '#fb923c' },
      { value: 40, label: 'Satisfactory', color: '#facc15' },
      { value: 0, label: 'Good', color: '#a3e635' },
    ],
    gradient: 'linear-gradient(to bottom, #9f1239 0%, #ef4444 25%, #fb923c 50%, #facc15 75%, #a3e635 100%)',
  },
  o3: {
    stops: [
      { value: 240, label: 'Severe', color: '#dc2626' },
      { value: 160, label: 'Very Poor', color: '#f97316' },
      { value: 100, label: 'Moderate', color: '#facc15' },
      { value: 60, label: 'Satisfactory', color: '#a3e635' },
      { value: 0, label: 'Good', color: '#5eead4' },
    ],
    gradient: 'linear-gradient(to bottom, #dc2626 0%, #f97316 25%, #facc15 50%, #a3e635 75%, #5eead4 100%)',
  },
  co: {
    stops: [
      { value: 4000, label: 'Severe', color: '#dc2626' },
      { value: 2000, label: 'Moderate', color: '#fb923c' },
      { value: 1000, label: 'Satisfactory', color: '#fde047' },
      { value: 500, label: 'Good', color: '#86efac' },
      { value: 0, label: 'Clean', color: '#67e8f9' },
    ],
    gradient: 'linear-gradient(to bottom, #dc2626 0%, #fb923c 25%, #fde047 50%, #86efac 75%, #67e8f9 100%)',
  },
};

interface ColorScaleProps {
  pollutant?: string;
  units?: string;
  measurementType?: string;
}

export default function ColorScale({
  pollutant = 'no2',
  units = 'µmol/m²',
  measurementType = 'satellite_column',
}: ColorScaleProps) {
  const isColumn = measurementType === 'satellite_column';
  const polKey = (pollutant || 'no2').toLowerCase();
  const config = POLLUTANT_COLOR_STOPS[polKey] || POLLUTANT_COLOR_STOPS.no2;
  const labelTitle = isColumn
    ? `${pollutant.toUpperCase()} Tropospheric Column`
    : `${pollutant.toUpperCase()} Surface Concentration`;

  return (
    <div className="absolute bottom-20 right-4 z-20 bg-[#11141d]/95 backdrop-blur-md border border-[#242938] rounded-xl p-3 shadow-2xl text-[#f1f3f7] select-none w-56">
      {/* Title */}
      <div className="text-[11px] font-semibold text-white tracking-wide uppercase font-mono mb-2 flex items-center justify-between border-b border-[#242938] pb-1.5">
        <span className="truncate">{labelTitle}</span>
        <span className="text-[10px] text-zinc-400 font-normal shrink-0 ml-1">{units}</span>
      </div>

      <div className="flex items-center gap-3">
        {/* Continuous Color Gradient Bar */}
        <div
          className="w-3.5 h-44 rounded-full border border-black/40 shadow-inner shrink-0"
          style={{ background: config.gradient }}
        />

        {/* Ticks & Category Labels */}
        <div className="flex-1 flex flex-col justify-between h-44 py-0.5 text-[10px]">
          {config.stops.map((stop) => (
            <div key={stop.value} className="flex items-center justify-between leading-none">
              <span className="font-mono text-zinc-300 font-bold w-10">{stop.value}</span>
              <span className="text-[10px] text-zinc-400 truncate text-right">{stop.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Advisory footnote */}
      <div className="mt-2 pt-1.5 border-t border-[#242938] text-[9px] text-zinc-400 text-center font-mono">
        {isColumn ? 'Satellite TROPOMI Column Density' : 'WHO / CPCB Regulatory Scale'}
      </div>
    </div>
  );
}
