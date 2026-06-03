import { useState, useEffect } from 'react';
import axios from 'axios';
import { MapComponent } from './components/Map';
import { RiskGauges } from './components/RiskGauges';
import {
  Wind, Search, Upload, FileText, MapPin,
  AlertTriangle, X, ChevronRight,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8001/api';

const ADAPTATION_LEVELS = [
  { key: 'No Adaptation',     label: 'Baseline' },
  { key: 'Medium Adaptation', label: 'Medium'   },
  { key: 'High Adaptation',   label: 'High'     },
];

const ZONE_CONFIG = {
  Red:    { label: 'High Risk',    color: '#ef4444', bg: 'rgba(239,68,68,0.12)'  },
  Orange: { label: 'Medium-High',  color: '#f97316', bg: 'rgba(249,115,22,0.12)' },
  Yellow: { label: 'Medium-Low',   color: '#eab308', bg: 'rgba(234,179,8,0.12)'  },
  Green:  { label: 'Low Risk',     color: '#22c55e', bg: 'rgba(34,197,94,0.12)'  },
};

// ─── Small helpers ────────────────────────────────────────────────────────────

const Spinner = ({ size = 14, color = 'white' }) => (
  <span style={{
    width: size, height: size, flexShrink: 0,
    border: `2px solid ${color}30`,
    borderTopColor: color,
    borderRadius: '50%',
    animation: 'spin 0.75s linear infinite',
    display: 'inline-block',
  }} />
);

const SectionLabel = ({ children }) => (
  <div style={{
    fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.12em',
    textTransform: 'uppercase', color: 'rgba(255,255,255,0.3)',
    marginBottom: '0.4rem',
  }}>
    {children}
  </div>
);

const Divider = () => (
  <div style={{ height: 1, background: 'rgba(255,255,255,0.05)', margin: '0.125rem 0' }} />
);

function ZoneBadge({ zone }) {
  const cfg = ZONE_CONFIG[zone] ?? ZONE_CONFIG.Green;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '0.3rem',
      padding: '0.2rem 0.55rem', borderRadius: 999,
      background: cfg.bg, color: cfg.color,
      border: `1px solid ${cfg.color}35`,
      fontSize: '0.68rem', fontWeight: 700, letterSpacing: '0.04em',
    }}>
      <span style={{ width: 5, height: 5, borderRadius: '50%', background: cfg.color }} />
      {cfg.label}
    </span>
  );
}

function RiskScoreRing({ score }) {
  const [anim, setAnim] = useState(0);
  useEffect(() => { const t = setTimeout(() => setAnim(score ?? 0), 80); return () => clearTimeout(t); }, [score]);

  const color = anim >= 90 ? '#ef4444' : anim >= 70 ? '#f97316' : anim >= 30 ? '#eab308' : '#22c55e';
  const r = 28, circ = 2 * Math.PI * r;

  return (
    <div style={{ position: 'relative', width: 80, height: 80, flexShrink: 0 }}>
      <svg width="80" height="80" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="40" cy="40" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" />
        <circle cx="40" cy="40" r={r} fill="none" stroke={color} strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${(anim / 100) * circ} ${circ}`}
          style={{ transition: 'stroke-dasharray 0.9s cubic-bezier(0.16,1,0.3,1), stroke 0.4s' }}
        />
      </svg>
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{ fontSize: '1.2rem', fontWeight: 900, color, lineHeight: 1 }}>{anim.toFixed(0)}</span>
        <span style={{ fontSize: '0.5rem', color: 'rgba(255,255,255,0.3)', fontWeight: 600, marginTop: 1 }}>/ 100</span>
      </div>
    </div>
  );
}

function ResultCard({ result, adaptationLevel }) {
  const key = adaptationLevel;
  const score    = result.klawa?.[key];
  const zone     = result.risk_zone?.[key];
  const windSpd  = result.wind_speed?.[key];
  const sc100    = result.scenarios?.find(s => s.rp === 100);
  const hasAsset = sc100 && Object.values(result.aal ?? {}).some(v => v != null);

  return (
    <div className="fade-in" style={{
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: 12, padding: '1rem',
      display: 'flex', flexDirection: 'column', gap: '0.875rem',
    }}>
      {/* Address */}
      <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'flex-start' }}>
        <MapPin size={12} style={{ color: 'var(--accent-light)', flexShrink: 0, marginTop: 2 }} />
        <span style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.55)', lineHeight: 1.45 }}>
          {result.address}
        </span>
      </div>

      {/* Score row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <RiskScoreRing score={score ?? 0} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
          <SectionLabel>Risk Index</SectionLabel>
          <ZoneBadge zone={zone} />
          {windSpd != null && (
            <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.45)', marginTop: 2 }}>
              Wind speed&nbsp;
              <span style={{ color: 'white', fontWeight: 700 }}>{windSpd.toFixed(1)} m/s</span>
            </div>
          )}
        </div>
      </div>

      {/* Gauges */}
      <RiskGauges data={result} />

      {/* Damage table */}
      {sc100 && (
        <div>
          <SectionLabel>Damage · 100-yr Return Period</SectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.22rem' }}>
            {['weak', 'moderate', 'strong'].map(res => {
              const dr  = sc100[`dr_${res}`];
              const aal = result.aal?.[res.charAt(0).toUpperCase() + res.slice(1)];
              return (
                <div key={res} style={{
                  display: 'flex', alignItems: 'center', gap: '0.5rem',
                  padding: '0.3rem 0.5rem', borderRadius: 6,
                  background: 'rgba(0,0,0,0.2)',
                }}>
                  <span style={{ fontSize: '0.68rem', fontWeight: 600, color: 'rgba(255,255,255,0.5)', width: 52, flexShrink: 0, textTransform: 'capitalize' }}>{res}</span>
                  <div style={{ flex: 1, height: 4, borderRadius: 99, background: 'rgba(255,255,255,0.07)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${(dr ?? 0) * 100}%`, background: 'var(--accent-light)', borderRadius: 99, transition: 'width 0.9s ease' }} />
                  </div>
                  <span style={{ fontSize: '0.68rem', fontWeight: 700, color: 'rgba(255,255,255,0.8)', width: 38, textAlign: 'right', flexShrink: 0 }}>
                    {dr != null ? (dr * 100).toFixed(1) + '%' : '—'}
                  </span>
                  {hasAsset && (
                    <span style={{ fontSize: '0.62rem', color: 'rgba(255,255,255,0.3)', width: 68, textAlign: 'right', flexShrink: 0 }}>
                      {aal != null ? aal.toLocaleString('en-NO', { maximumFractionDigits: 0 }) + ' kr' : '—'}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function BatchSummary({ data, onDownload }) {
  const highRisk = data.results.filter(r => parseFloat(r['Risk Index (No Adapt)']) > 80).length;
  const stats = [
    { label: 'Assets',     value: data.results.length,                      warn: false },
    { label: 'High Risk',  value: highRisk,                                  warn: highRisk > 0 },
    { label: 'NOK (M)',    value: (data.total_value / 1e6).toFixed(1) + 'M', warn: false },
  ];
  return (
    <div className="fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.4rem' }}>
        {stats.map(({ label, value, warn }) => (
          <div key={label} style={{
            padding: '0.5rem 0.25rem', borderRadius: 8, textAlign: 'center',
            background: warn ? 'rgba(239,68,68,0.08)' : 'rgba(0,0,0,0.3)',
            border: `1px solid ${warn ? 'rgba(239,68,68,0.25)' : 'rgba(255,255,255,0.06)'}`,
          }}>
            <div style={{ fontSize: '1rem', fontWeight: 800, color: warn ? '#ef4444' : 'white' }}>{value}</div>
            <div style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.07em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.35)', marginTop: 2 }}>{label}</div>
          </div>
        ))}
      </div>
      <button onClick={onDownload} style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem',
        width: '100%', padding: '0.6rem', borderRadius: 8, border: '1px solid rgba(34,197,94,0.3)',
        background: 'rgba(34,197,94,0.08)', color: '#22c55e',
        fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer',
        transition: 'background 0.2s',
      }}
        onMouseEnter={e => e.currentTarget.style.background = 'rgba(34,197,94,0.14)'}
        onMouseLeave={e => e.currentTarget.style.background = 'rgba(34,197,94,0.08)'}
      >
        <FileText size={13} /> Download Risk Report (.docx)
      </button>
    </div>
  );
}

function MapLegend() {
  return (
    <div style={{
      position: 'absolute', bottom: 20, right: 20,
      background: 'rgba(10,12,16,0.92)', backdropFilter: 'blur(14px)',
      border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: 10, padding: '0.7rem 0.875rem',
      minWidth: 130,
    }}>
      <div style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
        Risk Zones
      </div>
      {[
        ['#22c55e', 'Low'],
        ['#eab308', 'Medium-Low'],
        ['#f97316', 'Medium-High'],
        ['#ef4444', 'High'],
      ].map(([color, label]) => (
        <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
          <div style={{ width: 9, height: 9, borderRadius: 2, background: color, flexShrink: 0 }} />
          <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.65)' }}>{label}</span>
        </div>
      ))}
    </div>
  );
}

function MapLoadingOverlay() {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      height: '100%', gap: '1rem',
      background: 'radial-gradient(ellipse at 50% 40%, #0f172a, #0a0c10)',
    }}>
      <div style={{ position: 'relative', width: 52, height: 52 }}>
        <Wind size={52} color="rgba(39,85,164,0.25)" />
        <span style={{
          position: 'absolute', inset: 0, margin: 'auto',
          width: 52, height: 52, borderRadius: '50%',
          border: '2px solid transparent',
          borderTopColor: 'var(--accent-light)',
          animation: 'spin 1s linear infinite',
        }} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.25rem' }}>
        <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'rgba(255,255,255,0.7)' }}>Loading building data…</span>
        <span style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.3)' }}>This takes a few seconds on first load</span>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [mapData,        setMapData]        = useState(null);
  const [adaptationLevel, setAdaptation]    = useState('No Adaptation');
  const [searchAddress,  setSearchAddress]  = useState('');
  const [assetValue,     setAssetValue]     = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [batchResults,   setBatchResults]   = useState(null);
  const [loading,        setLoading]        = useState(false);
  const [error,          setError]          = useState(null);

  useEffect(() => { fetchMapData(); }, []);

  const fetchMapData = async () => {
    try {
      const cached = sessionStorage.getItem('windrisk_map_data');
      if (cached) { setMapData(JSON.parse(cached)); return; }
      const res = await axios.get(`${API_BASE}/map_data`);
      setMapData(res.data);
      try { sessionStorage.setItem('windrisk_map_data', JSON.stringify(res.data)); } catch { /* quota */ }
    } catch (err) { console.error(err); }
  };

  const handleAnalyze = async () => {
    if (!searchAddress) return;
    setLoading(true); setError(null); setAnalysisResult(null); setBatchResults(null);
    try {
      let url = `${API_BASE}/analyze?address=${encodeURIComponent(searchAddress)}`;
      if (assetValue) url += `&asset_value=${assetValue}`;
      const res = await axios.get(url);
      setAnalysisResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Address not found. Try a more specific Bergen address.');
    } finally { setLoading(false); }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    setLoading(true); setError(null); setAnalysisResult(null); setBatchResults(null);
    try {
      const res = await axios.post(`${API_BASE}/batch`, fd, { headers: { 'Content-Type': 'multipart/form-data' } });
      setBatchResults(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally { setLoading(false); }
  };

  const downloadReport = async () => {
    if (!batchResults) return;
    try {
      const res = await axios.post(`${API_BASE}/report`, batchResults, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const a = document.createElement('a');
      a.href = url; a.download = 'Portfolio_Wind_Risk_Report.docx';
      document.body.appendChild(a); a.click(); a.remove();
    } catch { setError('Failed to download report'); }
  };

  const analysing = loading && !batchResults;
  const uploading = loading && !analysisResult;

  return (
    <div style={{ display: 'flex', width: '100%', height: '100%' }}>

      {/* ── Sidebar ── */}
      <aside style={{
        width: 340, flexShrink: 0,
        display: 'flex', flexDirection: 'column',
        background: 'rgba(10,12,16,0.97)',
        borderRight: '1px solid rgba(255,255,255,0.055)',
        backdropFilter: 'blur(24px)',
        zIndex: 10,
      }}>

        {/* Header */}
        <div style={{
          padding: '1.1rem 1.25rem 1rem',
          borderBottom: '1px solid rgba(255,255,255,0.055)',
          display: 'flex', alignItems: 'center', gap: '0.75rem', flexShrink: 0,
        }}>
          <div style={{
            width: 36, height: 36, borderRadius: 9,
            background: 'linear-gradient(135deg, #1e3a6e 0%, #2755a4 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 18px rgba(39,85,164,0.45)',
            flexShrink: 0,
          }}>
            <Wind size={17} color="white" />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: '0.95rem', fontWeight: 800, color: 'white', letterSpacing: '-0.01em' }}>WindRisk</div>
            <div style={{ fontSize: '0.62rem', color: 'rgba(255,255,255,0.35)', marginTop: 1 }}>Bergen, Norway · Risk Assessment</div>
          </div>
          <span style={{
            fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
            padding: '0.15rem 0.45rem', borderRadius: 999,
            border: '1px solid rgba(59,130,246,0.35)', color: '#3b82f6',
          }}>Beta</span>
        </div>

        {/* Scrollable content */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>

          {/* Adaptation Scenario */}
          <div>
            <SectionLabel>Adaptation Scenario</SectionLabel>
            <div style={{
              display: 'flex', background: 'rgba(0,0,0,0.35)', borderRadius: 8, padding: 3,
              border: '1px solid rgba(255,255,255,0.06)',
            }}>
              {ADAPTATION_LEVELS.map(({ key, label }) => (
                <button key={key} onClick={() => setAdaptation(key)} style={{
                  flex: 1, padding: '0.38rem 0', borderRadius: 6, border: 'none', cursor: 'pointer',
                  fontSize: '0.74rem', fontWeight: 600, letterSpacing: '0.01em',
                  transition: 'all 0.18s',
                  background: adaptationLevel === key
                    ? 'linear-gradient(135deg, #1e3a6e, #2755a4)'
                    : 'transparent',
                  color: adaptationLevel === key ? 'white' : 'rgba(255,255,255,0.38)',
                  boxShadow: adaptationLevel === key ? '0 2px 10px rgba(39,85,164,0.4)' : 'none',
                }}>
                  {label}
                </button>
              ))}
            </div>
          </div>

          <Divider />

          {/* Single Asset */}
          <div>
            <SectionLabel>Single Asset Analysis</SectionLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>

              {/* Address input */}
              <div style={{ position: 'relative' }}>
                <MapPin size={13} style={{ position: 'absolute', left: 9, top: '50%', transform: 'translateY(-50%)', color: 'rgba(255,255,255,0.28)', pointerEvents: 'none' }} />
                <input type="text" placeholder="Enter Bergen address…" value={searchAddress}
                  onChange={e => setSearchAddress(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleAnalyze()}
                  style={{ paddingLeft: '1.85rem' }} />
              </div>

              {/* Asset value */}
              <div style={{ position: 'relative' }}>
                <span style={{ position: 'absolute', left: 9, top: '50%', transform: 'translateY(-50%)', fontSize: '0.68rem', fontWeight: 700, color: 'rgba(255,255,255,0.28)', pointerEvents: 'none' }}>NOK</span>
                <input type="number" placeholder="Asset value (Million)" value={assetValue}
                  onChange={e => setAssetValue(e.target.value)}
                  style={{ paddingLeft: '2.4rem' }} />
              </div>

              {/* Analyze button */}
              <button onClick={handleAnalyze} disabled={analysing || !searchAddress} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.45rem',
                padding: '0.62rem', borderRadius: 8, border: 'none', cursor: analysing || !searchAddress ? 'not-allowed' : 'pointer',
                background: analysing || !searchAddress
                  ? 'rgba(255,255,255,0.07)'
                  : 'linear-gradient(135deg, #1e3a6e, #2755a4)',
                color: analysing || !searchAddress ? 'rgba(255,255,255,0.35)' : 'white',
                fontSize: '0.8rem', fontWeight: 700, width: '100%',
                boxShadow: analysing || !searchAddress ? 'none' : '0 0 18px rgba(39,85,164,0.35)',
                transition: 'all 0.2s',
              }}>
                {analysing ? <><Spinner /> Analyzing…</> : <><Search size={13} /> Analyze Asset</>}
              </button>
            </div>
          </div>

          {/* Result card */}
          {analysisResult && (
            <ResultCard result={analysisResult} adaptationLevel={adaptationLevel} />
          )}

          <Divider />

          {/* Batch */}
          <div>
            <SectionLabel>Portfolio Batch Analysis</SectionLabel>
            <label style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.35rem',
              padding: '0.875rem 1rem', borderRadius: 8, cursor: 'pointer',
              border: '1px dashed rgba(59,130,246,0.28)',
              background: 'rgba(59,130,246,0.04)',
              transition: 'background 0.2s',
            }}
              onMouseEnter={e => e.currentTarget.style.background = 'rgba(59,130,246,0.09)'}
              onMouseLeave={e => e.currentTarget.style.background = 'rgba(59,130,246,0.04)'}
            >
              {uploading
                ? <Spinner size={18} color="#3b82f6" />
                : <Upload size={17} color="#3b82f6" />
              }
              <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#3b82f6' }}>
                {uploading ? 'Processing portfolio…' : 'Upload Excel (.xlsx)'}
              </span>
              <span style={{ fontSize: '0.62rem', color: 'rgba(255,255,255,0.28)', textAlign: 'center', lineHeight: 1.4 }}>
                Requires columns: Address,<br />Monetary Asset Value (million NOK)
              </span>
              <input type="file" accept=".xlsx" style={{ display: 'none' }} onChange={handleFileUpload} />
            </label>

            {batchResults && <div style={{ marginTop: '0.6rem' }}><BatchSummary data={batchResults} onDownload={downloadReport} /></div>}
          </div>

          {/* Error banner */}
          {error && (
            <div className="fade-in" style={{
              display: 'flex', alignItems: 'flex-start', gap: '0.5rem',
              padding: '0.7rem 0.75rem', borderRadius: 8,
              background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)',
            }}>
              <AlertTriangle size={13} color="#ef4444" style={{ flexShrink: 0, marginTop: 1 }} />
              <span style={{ flex: 1, fontSize: '0.75rem', color: '#f87171', lineHeight: 1.45 }}>{error}</span>
              <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(239,68,68,0.5)', padding: 0, flexShrink: 0, display: 'flex' }}>
                <X size={13} />
              </button>
            </div>
          )}

          {/* Empty state hint */}
          {!analysisResult && !batchResults && !loading && !error && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              padding: '0.6rem 0.75rem', borderRadius: 8,
              background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)',
            }}>
              <ChevronRight size={12} color="rgba(255,255,255,0.2)" />
              <span style={{ fontSize: '0.7rem', color: 'rgba(255,255,255,0.28)', lineHeight: 1.45 }}>
                Enter an address above or click a building on the map to assess wind risk.
              </span>
            </div>
          )}

        </div>
      </aside>

      {/* ── Map ── */}
      <main style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {mapData
          ? <MapComponent data={mapData} adaptationLevel={adaptationLevel} />
          : <MapLoadingOverlay />
        }
        {mapData && <MapLegend />}
      </main>

    </div>
  );
}
