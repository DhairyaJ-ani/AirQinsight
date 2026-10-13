# AirQ Insight — Frontend Web Application

The interactive geospatial web application for **AirQ Insight**, built with Next.js 16 (React 19), Tailwind CSS 4, MapLibre GL, and Three.js.

*Companion Documentation: [Repository README](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/README.md) · [System Architecture](file:///Users/dhairyapunitjani/Development/My%20bharat%202026/AirQinsight/SYSTEM_ARCHITECTURE.md)*

---

## 🚀 Key Modules & Pages

- **`/` (Geospatial Map):** Interactive national and regional air quality canvas with real-time layer controls and pollutant selection.
- **`/visualization` (Plume Flow & Point Forecasting):** High-resolution raster inspection, interactive timeline scrubber, dynamic color legends, and coordinate-specific point forecasting.
- **`/simulator` (What-If Policy Simulator):** Digital twin counterfactual scenario simulator for testing emissions abatement policies (traffic reduction, industrial limits, construction halts).
- **`/drone` (Drone Flight Planning Portal):** Autonomous mission control generating wind-compensated UAV inspection flight waypoints over detected hotspots.
- **`/chatbot` (AI Multi-Agent Assistant):** Natural language environmental intelligence interface rendering dynamic **Mission Cards** powered by LangGraph.
- **`/reports` (Regulatory Compliance Center):** Interactive CPCB NAAQS regulatory compliance dashboard with one-click official PDF report synthesis.
- **`/globe` (3D Atmospheric Globe):** Three.js-powered WebGL planetary globe for transboundary air quality monitoring.
- **`/upload` (Raster Ingestion Portal):** Quality control and ingestion portal for user-uploaded GeoTIFF satellite files.

---

## 🛠️ Getting Started

### Prerequisites
- **Node.js:** `>= 18.18` (Node 20+ LTS recommended)
- **npm:** `>= 9.0`

### Installation
```bash
npm install
```

### Environment Configuration
Copy `.env.example` to `.env.local`:
```bash
cp .env.example .env.local
```

Configure your environment variables:
```env
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your-google-maps-api-key
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key
```
*(For local offline evaluation, `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY` is pre-configured with the default key).*

### Running the Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production
```bash
npm run build
npm run start
```
*(Production builds are optimized with Turbopack and pre-render 17 static and dynamic route pages).*
