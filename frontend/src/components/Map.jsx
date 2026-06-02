import DeckGL from '@deck.gl/react';
import { GeoJsonLayer } from '@deck.gl/layers';
import Map from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

const INITIAL_VIEW_STATE = {
  longitude: 5.3221,
  latitude: 60.3913,
  zoom: 12,
  pitch: 0,
  bearing: 0
};

const ADAPTER_KEY = {
  'No Adaptation':     'no_adapt',
  'Medium Adaptation': 'medium_adapt',
  'High Adaptation':   'high_adapt'
};

export const MapComponent = ({ data, adaptationLevel }) => {
  const adapterKey = ADAPTER_KEY[adaptationLevel] ?? 'no_adapt';

  const layers = [
    new GeoJsonLayer({
      id: 'buildings-layer',
      data,
      pickable: true,
      stroked: true,
      filled: true,
      getFillColor: f => f.properties?.[`risk_color_rgba_${adapterKey}`] ?? [128, 128, 128, 0],
      getLineColor: [255, 255, 255, 20],
      lineWidthMinPixels: 1,
      updateTriggers: { getFillColor: [adapterKey] }
    })
  ];

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', borderRadius: '12px', overflow: 'hidden' }}>
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        controller={true}
        layers={layers}
        getTooltip={({ object }) => object && {
          html: `<div style="font-family:'Inter',sans-serif;padding:4px;"><b>Building ID:</b> ${object.id}</div>`,
          style: {
            backgroundColor: 'var(--bg-primary)',
            color: 'var(--text-primary)',
            borderRadius: '8px',
            border: '1px solid var(--glass-border)',
            padding: '8px'
          }
        }}
      >
        <Map mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" />
      </DeckGL>
    </div>
  );
};
