import { useEffect, useState } from 'react';

const Gauge = ({ label, value }) => {
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(value ?? 0), 80);
    return () => clearTimeout(t);
  }, [value]);

  const color = animated >= 90 ? '#ef4444'
              : animated >= 70 ? '#f97316'
              : animated >= 30 ? '#eab308'
              : '#22c55e';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <span style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.45)', width: 56, flexShrink: 0, fontWeight: 500 }}>
        {label}
      </span>
      <div style={{ flex: 1, height: 5, borderRadius: 99, background: 'rgba(255,255,255,0.07)', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${animated}%`,
          background: color,
          borderRadius: 99,
          transition: 'width 0.9s cubic-bezier(0.16,1,0.3,1)',
          boxShadow: `0 0 6px ${color}50`,
        }} />
      </div>
      <span style={{ fontSize: '0.72rem', fontWeight: 700, color, width: 30, textAlign: 'right', flexShrink: 0 }}>
        {animated.toFixed(0)}
      </span>
    </div>
  );
};

export const RiskGauges = ({ data }) => {
  if (!data?.klawa) return null;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
      <span style={{ fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.3)', marginBottom: 2 }}>
        Risk Score by Scenario
      </span>
      <Gauge label="Baseline" value={data.klawa['No Adaptation']}     />
      <Gauge label="Medium"   value={data.klawa['Medium Adaptation']} />
      <Gauge label="High"     value={data.klawa['High Adaptation']}   />
    </div>
  );
};
