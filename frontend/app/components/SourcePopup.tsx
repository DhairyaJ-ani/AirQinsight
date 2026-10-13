'use client';

import React, { useEffect, useState, useCallback } from 'react';
import {
  X,
  Factory,
  Zap,
  Car,
  HardHat,
  HelpCircle,
  AlertTriangle,
  Info,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Loader2,
} from 'lucide-react';
import { POLLUTANT_DEFS } from './PollutantSelector';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

function authHeaders(): Record<string, string> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

interface SourceAssessment {
  source_id?: string;
  source_name: string;
  source_category: string;
  identification_status: string;
  selected_pollutant: string;
  pollutant_display: string;
  observation_date: string;
  source_provenance: string;
  local_value: number | null;
  local_units: string;
  measurement_type: string;
  in_raster_extent: boolean;
  surrounding_statistics: {
    mean?: number;
    median?: number;
    max?: number;
    min?: number;
    std?: number;
    valid_pixel_count?: number;
    valid_pixel_fraction?: number;
    note?: string;
  };
  analysis_radius_km: number;
  agent_interpretation: {
    attribution_statement: string;
    confidence_level: string;
    evidence: string[];
    alternative_explanations: string[];
    limitations: string[];
    recommended_follow_up: string[];
    measurement_metadata: {
      measurement_type: string;
      units: string;
      source_dataset: string;
      spatial_resolution_km: number;
      coarse_cell_note: string;
    };
  };
  data_availability: string;
  data_availability_note?: string;
}

interface SourcePopupProps {
  sourceId: string | null;
  sourceName: string;
  sourceCategory: string;
  sourceLat: number;
  sourceLon: number;
  emissionFactor: number;
  sourceDetails: string;
  selectedPollutant: string;
  selectedDate: string;
  onClose: () => void;
}

function CategoryIcon({ category, color }: { category: string; color: string }) {
  const cls = `w-4 h-4`;
  if (category === 'POWER_PLANT') return <Zap className={cls} style={{ color }} />;
  if (category === 'FACTORY') return <Factory className={cls} style={{ color }} />;
  if (category === 'TRAFFIC_CORRIDOR') return <Car className={cls} style={{ color }} />;
  if (category === 'CONSTRUCTION_SITE') return <HardHat className={cls} style={{ color }} />;
  return <HelpCircle className={cls} style={{ color }} />;
}

function ConfidenceChip({ level }: { level: string }) {
  const map: Record<string, { color: string; bg: string; border: string }> = {
    high: { color: '#4ade80', bg: '#14532d22', border: '#4ade8044' },
    'low-moderate': { color: '#fbbf24', bg: '#78350f22', border: '#fbbf2444' },
    moderate: { color: '#fbbf24', bg: '#78350f22', border: '#fbbf2444' },
    low: { color: '#94a3b8', bg: '#1e293b', border: '#334155' },
    insufficient: { color: '#f87171', bg: '#7f1d1d22', border: '#f8717144' },
  };
  const style = map[level.toLowerCase()] ?? map.low;
  return (
    <span
      className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded border"
      style={{ color: style.color, backgroundColor: style.bg, borderColor: style.border }}
    >
      {level.toUpperCase()} CONFIDENCE
    </span>
  );
}

function DataBadge({ availability, note }: { availability: string; note?: string }) {
  if (availability === 'available')
    return <span className="text-[10px] font-mono text-emerald-400">● DATA AVAILABLE</span>;
  if (availability === 'nodata')
    return (
      <span className="text-[10px] font-mono text-amber-400" title={note}>
        ● NO DATA (Cloud/Missing)
      </span>
    );
  return (
    <span className="text-[10px] font-mono text-zinc-500" title={note}>
      ● OUT OF EXTENT
    </span>
  );
}

function categoryColor(category: string): string {
  const map: Record<string, string> = {
    POWER_PLANT: '#f43f5e',
    FACTORY: '#fbbf24',
    TRAFFIC_CORRIDOR: '#38bdf8',
    CONSTRUCTION_SITE: '#a78bfa',
  };
  return map[category] ?? '#94a3b8';
}

export default function SourcePopup({
  sourceName,
  sourceCategory,
  sourceLat,
  sourceLon,
  emissionFactor,
  sourceDetails,
  selectedPollutant,
  selectedDate,
  onClose,
}: SourcePopupProps) {
  const [assessment, setAssessment] = useState<SourceAssessment | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evidenceExpanded, setEvidenceExpanded] = useState(false);
  const [limitationsExpanded, setLimitationsExpanded] = useState(false);

  const pol = POLLUTANT_DEFS.find((p) => p.key === selectedPollutant) ?? POLLUTANT_DEFS[0];

  const fetchAssessment = useCallback(async () => {
    if (!selectedDate || !selectedPollutant) return;
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        lat: sourceLat.toFixed(5),
        lon: sourceLon.toFixed(5),
        pollutant: selectedPollutant,
        date: selectedDate,
        source_name: sourceName,
        source_category: sourceCategory,
        radius_km: '5.0',
        identification_status: 'candidate_source',
      });
      const res = await fetch(`${API_BASE}/api/v1/analyze/source-assessment?${params}`, {
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setAssessment(data);
    } catch (e) {
      setError(`Could not load analysis: ${e instanceof Error ? e.message : e}`);
    } finally {
      setLoading(false);
    }
  }, [sourceLat, sourceLon, selectedPollutant, selectedDate, sourceName, sourceCategory]);

  useEffect(() => {
    fetchAssessment();
  }, [fetchAssessment]);

  const accentColor = categoryColor(sourceCategory);

  return (
    <div
      id="source-popup-panel"
      className="absolute right-4 top-16 z-40 w-[360px] max-h-[calc(100vh-96px)] overflow-y-auto
        bg-[#0c0f17]/98 border border-[#1e2638] rounded-xl shadow-2xl backdrop-blur-md
        flex flex-col text-xs text-zinc-200 animate-fade-in"
      style={{ scrollbarWidth: 'thin' }}
    >
      {/* Header */}
      <div
        className="flex items-start justify-between px-4 pt-4 pb-3 border-b border-[#1e2638]"
        style={{ borderLeftColor: accentColor, borderLeftWidth: 3 }}
      >
        <div className="flex items-start gap-2 min-w-0 flex-1">
          <div
            className="w-7 h-7 rounded flex items-center justify-center shrink-0 mt-0.5"
            style={{ backgroundColor: `${accentColor}18`, border: `1px solid ${accentColor}44` }}
          >
            <CategoryIcon category={sourceCategory} color={accentColor} />
          </div>
          <div className="min-w-0">
            <div className="font-semibold text-zinc-100 leading-tight truncate" title={sourceName}>
              {sourceName}
            </div>
            <div className="flex items-center gap-2 mt-0.5 flex-wrap">
              <span
                className="text-[10px] font-mono px-1.5 py-0.5 rounded border"
                style={{ color: accentColor, backgroundColor: `${accentColor}18`, borderColor: `${accentColor}44` }}
              >
                {sourceCategory.replace(/_/g, ' ')}
              </span>
              <DataBadge
                availability={assessment?.data_availability ?? 'pending'}
                note={assessment?.data_availability_note}
              />
            </div>
          </div>
        </div>
        <button
          onClick={onClose}
          aria-label="Close panel"
          className="ml-2 p-1 text-zinc-500 hover:text-zinc-200 hover:bg-[#1e2638] rounded transition-colors cursor-pointer shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Source details */}
      <div className="px-4 py-3 border-b border-[#1e2638] space-y-1">
        <div className="text-zinc-400 text-[11px] leading-relaxed">{sourceDetails}</div>
        <div className="flex gap-3 mt-1 text-[10px] font-mono text-zinc-500">
          <span>
            Emission Factor:{' '}
            <span className="text-zinc-300 font-semibold">{(emissionFactor * 100).toFixed(0)}%</span>
          </span>
          <span>
            Coords:{' '}
            <span className="text-zinc-300">
              {sourceLat.toFixed(4)}°N, {sourceLon.toFixed(4)}°E
            </span>
          </span>
        </div>
      </div>

      {/* Pollutant value section */}
      <div className="px-4 py-3 border-b border-[#1e2638]">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
            Observed {pol.formula}
          </span>
          {loading && <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />}
        </div>

        {error && (
          <div className="flex items-center gap-1.5 text-amber-400 text-[11px] py-1">
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
            {error}
          </div>
        )}

        {assessment && !error && (
          <>
            {/* Local value */}
            <div className="flex items-end gap-2 mb-2">
              {assessment.local_value !== null ? (
                <>
                  <span
                    className="text-2xl font-bold font-mono tracking-tight"
                    style={{ color: pol.color }}
                  >
                    {assessment.local_value.toFixed(2)}
                  </span>
                  <span className="text-zinc-400 text-[11px] mb-1 font-mono">
                    {assessment.local_units}
                  </span>
                  <span className="text-zinc-500 text-[10px] mb-1">at source cell (~3.9 km)</span>
                </>
              ) : (
                <span className="text-zinc-500 text-[11px] italic">
                  No data at this pixel
                  {assessment.surrounding_statistics.note
                    ? ` — ${assessment.surrounding_statistics.note}`
                    : ''}
                </span>
              )}
            </div>

            {/* Surrounding stats grid */}
            {assessment.surrounding_statistics.mean !== undefined && (
              <div className="grid grid-cols-4 gap-1 bg-[#111827] rounded-lg p-2 mb-2">
                {([
                  { label: 'Mean', v: assessment.surrounding_statistics.mean },
                  { label: 'Median', v: assessment.surrounding_statistics.median },
                  { label: 'Max', v: assessment.surrounding_statistics.max },
                  { label: 'Std', v: assessment.surrounding_statistics.std },
                ] as const).map(({ label, v }) => (
                  <div key={label} className="flex flex-col items-center gap-0.5">
                    <span className="text-[9px] font-mono text-zinc-500 uppercase">{label}</span>
                    <span className="text-[11px] font-mono text-zinc-200 font-semibold">
                      {v !== undefined ? v.toFixed(1) : '—'}
                    </span>
                  </div>
                ))}
              </div>
            )}
            {assessment.surrounding_statistics.valid_pixel_fraction !== undefined && (
              <div className="text-[10px] text-zinc-500 font-mono">
                Coverage:{' '}
                <span className="text-zinc-300">
                  {(assessment.surrounding_statistics.valid_pixel_fraction * 100).toFixed(0)}%
                </span>{' '}
                valid pixels within {assessment.analysis_radius_km} km radius ·{' '}
                <span className="text-zinc-400">{assessment.measurement_type.replace(/_/g, ' ')}</span>
              </div>
            )}
          </>
        )}

        {!loading && !assessment && !error && (
          <div className="text-zinc-500 text-[11px] italic">Awaiting date selection…</div>
        )}
      </div>

      {/* Analysis Agent interpretation */}
      {assessment?.agent_interpretation && (
        <div className="px-4 py-3 border-b border-[#1e2638] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-1">
              <Info className="w-3.5 h-3.5 text-blue-400" />
              Analysis Agent
            </span>
            <ConfidenceChip level={assessment.agent_interpretation.confidence_level} />
          </div>

          <p className="text-[11px] text-zinc-300 leading-relaxed">
            {assessment.agent_interpretation.attribution_statement}
          </p>

          {/* Evidence chain — collapsible */}
          {assessment.agent_interpretation.evidence.length > 0 && (
            <div>
              <button
                onClick={() => setEvidenceExpanded((v) => !v)}
                className="flex items-center gap-1 text-[10px] text-blue-400 hover:text-blue-300 transition-colors cursor-pointer"
              >
                {evidenceExpanded ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
                Evidence chain ({assessment.agent_interpretation.evidence.length} items)
              </button>
              {evidenceExpanded && (
                <ul className="mt-1 space-y-0.5">
                  {assessment.agent_interpretation.evidence.map((e, i) => (
                    <li key={i} className="flex items-start gap-1 text-[10px] text-zinc-400">
                      <CheckCircle className="w-3 h-3 text-emerald-500 mt-0.5 shrink-0" />
                      {e}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* Alternative explanations */}
          {assessment.agent_interpretation.alternative_explanations.length > 0 && (
            <div>
              <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-0.5">
                Alternative explanations
              </div>
              <ul className="space-y-0.5">
                {assessment.agent_interpretation.alternative_explanations.map((e, i) => (
                  <li key={i} className="flex items-start gap-1 text-[10px] text-zinc-400">
                    <span className="text-zinc-600 shrink-0">•</span>
                    {e}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Limitations — collapsible */}
      {assessment?.agent_interpretation.limitations && (
        <div className="px-4 py-3 border-b border-[#1e2638]">
          <button
            onClick={() => setLimitationsExpanded((v) => !v)}
            className="flex items-center gap-1 text-[10px] text-amber-400 hover:text-amber-300 transition-colors cursor-pointer"
          >
            <AlertTriangle className="w-3 h-3" />
            Scientific limitations
            {limitationsExpanded ? (
              <ChevronUp className="w-3 h-3 ml-0.5" />
            ) : (
              <ChevronDown className="w-3 h-3 ml-0.5" />
            )}
          </button>
          {limitationsExpanded && (
            <ul className="mt-1.5 space-y-1">
              {assessment.agent_interpretation.limitations.map((l, i) => (
                <li key={i} className="flex items-start gap-1 text-[10px] text-zinc-400 leading-relaxed">
                  <AlertTriangle className="w-3 h-3 text-amber-500/60 mt-0.5 shrink-0" />
                  {l}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Dataset provenance */}
      {assessment && (
        <div className="px-4 py-3 text-[10px] text-zinc-500 font-mono leading-relaxed space-y-0.5">
          <div>
            Dataset: <span className="text-zinc-400">{assessment.source_provenance}</span>
          </div>
          <div>
            Date: <span className="text-zinc-400">{assessment.observation_date}</span>
          </div>
          <div>
            Resolution:{' '}
            <span className="text-zinc-400">
              ~{assessment.agent_interpretation.measurement_metadata.spatial_resolution_km} km grid cell
            </span>
          </div>
          <div className="pt-1 text-zinc-600 italic text-[9px]">
            {assessment.agent_interpretation.measurement_metadata.coarse_cell_note}
          </div>
        </div>
      )}
    </div>
  );
}
