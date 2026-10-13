'use client';

import React from 'react';
import { Wind } from 'lucide-react';

export interface PollutantDef {
  key: string;
  formula: string;
  name: string;
  units: string;
  /** CPCB 24-h / 8-h limit in native units (null = column / no direct NAAQS) */
  naaqs: number | null;
  /** colour used for the selector badge */
  color: string;
}

export const POLLUTANT_DEFS: PollutantDef[] = [
  { key: 'no2',  formula: 'NO₂',  name: 'Nitrogen Dioxide',      units: 'µmol/m²',  naaqs: null,  color: '#818cf8' },
  { key: 'pm25', formula: 'PM₂.₅', name: 'Fine Particulate Matter', units: 'µg/m³', naaqs: 60,    color: '#34d399' },
  { key: 'pm10', formula: 'PM₁₀',  name: 'Particulate Matter 10', units: 'µg/m³',   naaqs: 100,   color: '#fbbf24' },
  { key: 'so2',  formula: 'SO₂',   name: 'Sulfur Dioxide',        units: 'µg/m³',   naaqs: 80,    color: '#f87171' },
  { key: 'o3',   formula: 'O₃',    name: 'Ozone',                 units: 'µg/m³',   naaqs: 100,   color: '#38bdf8' },
  { key: 'co',   formula: 'CO',    name: 'Carbon Monoxide',       units: 'µg/m³',   naaqs: 2000,  color: '#e879f9' },
];

interface PollutantSelectorProps {
  selected: string;
  onChange: (pollutant: string) => void;
}

/**
 * Compact floating pollutant toggle bar that sits in the top-left bar of the Home page.
 * Renders one pill-button per pollutant; active one is highlighted with its accent colour.
 */
export default function PollutantSelector({ selected, onChange }: PollutantSelectorProps) {
  return (
    <div
      id="pollutant-selector"
      className="flex items-center gap-0.5 h-10 px-2 bg-[#141721]/95 border border-[#2e3547] rounded-md shadow-xl backdrop-blur-md shrink-0"
      role="group"
      aria-label="Pollutant selector"
    >
      <Wind className="w-3.5 h-3.5 text-zinc-400 shrink-0 mr-0.5" />
      {POLLUTANT_DEFS.map((pol) => {
        const isActive = selected === pol.key;
        return (
          <button
            key={pol.key}
            id={`pollutant-btn-${pol.key}`}
            onClick={() => onChange(pol.key)}
            title={`${pol.name} (${pol.units})`}
            aria-pressed={isActive}
            className={`relative px-2 py-1 rounded text-[11px] font-mono font-semibold transition-all duration-150 cursor-pointer select-none shrink-0 ${
              isActive
                ? 'text-white shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
            style={
              isActive
                ? {
                    backgroundColor: `${pol.color}22`,
                    border: `1px solid ${pol.color}88`,
                    color: pol.color,
                  }
                : { border: '1px solid transparent' }
            }
          >
            {pol.formula}
          </button>
        );
      })}
    </div>
  );
}
