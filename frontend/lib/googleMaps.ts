export const STORAGE_KEY_MAPS_KEY = 'aeroscale_google_maps_key';
export const STORAGE_KEY_MAPS_POOL = 'aeroscale_google_maps_pool';
export const STORAGE_KEY_AUTO_CYCLE = 'aeroscale_google_maps_autocycle';
export const STORAGE_KEY_FAILED_KEYS = 'aeroscale_failed_maps_keys';

/**
 * Returns an array of valid Google Maps API keys defined in environment variables.
 * Looks for NEXT_PUBLIC_GOOGLE_MAPS_API_KEY and NEXT_PUBLIC_GOOGLE_MAPS_FALLBACK_KEY.
 */
export function getEnvKeyPool(): string[] {
  const rawPrimary = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '';
  const rawFallback =
    process.env.NEXT_PUBLIC_GOOGLE_MAPS_FALLBACK_KEY ||
    process.env.NEXT_PUBLIC_GOOGLE_MAPS_BACKUP_KEY ||
    '';

  const extractKeys = (val: string) =>
    val
      .split(/[,;\s]+/)
      .map((k) => k.trim())
      .filter((k) => k.length > 0);

  const combined = [...extractKeys(rawPrimary), ...extractKeys(rawFallback)];
  return Array.from(new Set(combined));
}

/**
 * Retrieve keys marked as failed/exhausted in the current session.
 */
export function getFailedKeys(): string[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY_FAILED_KEYS);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Mark a key as failed so the application automatically selects an untried fallback key.
 */
export function markKeyAsFailed(key: string) {
  if (typeof window === 'undefined' || !key) return;
  try {
    const failed = getFailedKeys();
    if (!failed.includes(key)) {
      failed.push(key);
      sessionStorage.setItem(STORAGE_KEY_FAILED_KEYS, JSON.stringify(failed));
    }
  } catch {
    // ignore
  }
}

/**
 * Get active Google Maps API key, automatically preferring non-failed keys from the environment pool.
 */
export function getActiveGoogleMapsApiKey(): string {
  const pool = getEnvKeyPool();
  if (pool.length === 0) {
    return process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '';
  }

  if (typeof window !== 'undefined') {
    const failed = getFailedKeys();

    try {
      const stored = localStorage.getItem(STORAGE_KEY_MAPS_KEY);
      // Only honor stored key if it is present in the env pool and has not failed
      if (stored && pool.includes(stored.trim()) && !failed.includes(stored.trim())) {
        return stored.trim();
      }
    } catch {
      // ignore
    }

    // Pick first non-failed key from env pool (primary -> fallback)
    const available = pool.find((k) => !failed.includes(k));
    if (available) {
      return available;
    }
  }

  return pool[0] || process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || '';
}

export function getKeyPool(): string[] {
  const envPool = getEnvKeyPool();
  if (typeof window === 'undefined') return envPool;
  try {
    const raw = localStorage.getItem(STORAGE_KEY_MAPS_POOL);
    if (!raw) return envPool;
    const parsed = JSON.parse(raw);
    const storedPool = Array.isArray(parsed)
      ? parsed.filter((k: any) => typeof k === 'string' && k.trim().length > 0)
      : [];
    return Array.from(new Set([...envPool, ...storedPool]));
  } catch {
    return envPool;
  }
}

export function setKeyPool(keys: string[]) {
  if (typeof window === 'undefined') return;
  const cleaned = keys.map((k) => k.trim()).filter((k) => k.length > 0);
  localStorage.setItem(STORAGE_KEY_MAPS_POOL, JSON.stringify(cleaned));
  window.dispatchEvent(new CustomEvent('aeroscale:maps_pool_updated', { detail: { pool: cleaned } }));
}

export function setActiveKey(key: string) {
  if (typeof window === 'undefined') return;
  const trimmed = key.trim();
  localStorage.setItem(STORAGE_KEY_MAPS_KEY, trimmed);
  window.dispatchEvent(new CustomEvent('aeroscale:maps_key_changed', { detail: { key: trimmed } }));
}

export function cycleToNextKey(): { nextKey: string | null; index: number; total: number } {
  if (typeof window === 'undefined') return { nextKey: null, index: -1, total: 0 };
  const pool = getKeyPool();
  if (pool.length === 0) return { nextKey: null, index: -1, total: 0 };

  const current = getActiveGoogleMapsApiKey();
  const currentIndex = pool.findIndex((k) => k === current);
  const nextIndex = (currentIndex + 1) % pool.length;
  const nextKey = pool[nextIndex];

  setActiveKey(nextKey);
  return { nextKey, index: nextIndex, total: pool.length };
}

export const GOOGLE_MAPS_API_KEY = getActiveGoogleMapsApiKey();

let googleMapsPromise: Promise<any> | null = null;

export function resetGoogleMapsSdk() {
  googleMapsPromise = null;
  if (typeof document !== 'undefined') {
    const existing = document.querySelector('script[data-google-maps-script="true"]');
    if (existing && existing.parentNode) {
      existing.parentNode.removeChild(existing);
    }
  }
}

export function loadGoogleMaps(): Promise<any> {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Window is not defined'));
  }

  // Already loaded
  if ((window as any).google && (window as any).google.maps) {
    return Promise.resolve((window as any).google.maps);
  }

  // In-flight loading promise
  if (googleMapsPromise) {
    return googleMapsPromise;
  }

  const activeKey = getActiveGoogleMapsApiKey();

  googleMapsPromise = new Promise((resolve, reject) => {
    // Check if script element already exists in document
    const existingScript = document.querySelector('script[data-google-maps-script="true"]');
    if (existingScript) {
      existingScript.addEventListener('load', () => {
        resolve((window as any).google?.maps);
      });
      existingScript.addEventListener('error', (err) => {
        reject(err);
      });
      return;
    }

    // Global Google Maps Auth Failure hook
    (window as any).gm_authFailure = () => {
      console.warn(
        '[GoogleMaps] Authentication failed for key:',
        activeKey ? `${activeKey.slice(0, 8)}...` : '(none)'
      );
      markKeyAsFailed(activeKey);
      window.dispatchEvent(new CustomEvent('aeroscale:maps_auth_failure', { detail: { key: activeKey } }));

      // Fallback functionality: check if an untried fallback key is configured
      const pool = getKeyPool();
      const failed = getFailedKeys();
      const nextKey = pool.find((k) => !failed.includes(k));

      if (nextKey && nextKey !== activeKey) {
        console.warn(
          `[GoogleMaps] Switching to environment fallback key: ${nextKey.slice(0, 8)}... and reloading`
        );
        setActiveKey(nextKey);
        setTimeout(() => {
          window.location.reload();
        }, 500);
      } else {
        console.error(
          '[GoogleMaps] All configured Google Maps API keys (including fallback) have failed or exhausted quota.'
        );
      }
    };

    const script = document.createElement('script');
    script.setAttribute('data-google-maps-script', 'true');
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(
      activeKey
    )}&libraries=places,geometry`;
    script.async = true;
    script.defer = true;

    script.onload = () => {
      if ((window as any).google && (window as any).google.maps) {
        resolve((window as any).google.maps);
      } else {
        reject(new Error('Google Maps SDK loaded but window.google.maps is undefined'));
      }
    };

    script.onerror = (error) => {
      console.warn(
        `[GoogleMaps] Failed to load script with key ${activeKey ? activeKey.slice(0, 8) + '...' : '(none)'}`
      );
      markKeyAsFailed(activeKey);
      resetGoogleMapsSdk();

      // Check if fallback key is available in the pool
      const pool = getKeyPool();
      const failed = getFailedKeys();
      const nextKey = pool.find((k) => !failed.includes(k));

      if (nextKey && nextKey !== activeKey) {
        console.warn(
          `[GoogleMaps] Automatically retrying load with fallback key: ${nextKey.slice(0, 8)}...`
        );
        setActiveKey(nextKey);
        loadGoogleMaps().then(resolve).catch(reject);
        return;
      }

      window.dispatchEvent(new CustomEvent('aeroscale:maps_auth_failure', { detail: { key: activeKey } }));
      reject(new Error(`Failed to load Google Maps SDK: ${error}`));
    };

    document.head.appendChild(script);
  });

  return googleMapsPromise;
}


// Ultra-crisp cyber dark theme tailored for AeroScale's high-tech UI
export const DARK_MAP_STYLES: any[] = [
  { elementType: 'geometry', stylers: [{ color: '#10131c' }] },
  { elementType: 'labels.text.stroke', stylers: [{ color: '#090b10' }, { weight: 3 }] },
  { elementType: 'labels.text.fill', stylers: [{ color: '#94a3b8' }] },
  {
    featureType: 'administrative',
    elementType: 'geometry',
    stylers: [{ visibility: 'on' }, { color: '#273145' }],
  },
  {
    featureType: 'administrative.country',
    elementType: 'geometry.stroke',
    stylers: [{ color: '#3b82f6' }, { weight: 1.2 }, { opacity: 0.35 }],
  },
  {
    featureType: 'administrative.locality',
    elementType: 'labels.text.fill',
    stylers: [{ color: '#e2e8f0' }, { weight: 500 }],
  },
  {
    featureType: 'poi',
    elementType: 'labels.text.fill',
    stylers: [{ color: '#64748b' }],
  },
  {
    featureType: 'poi',
    elementType: 'labels.icon',
    stylers: [{ visibility: 'off' }],
  },
  {
    featureType: 'poi.park',
    elementType: 'geometry',
    stylers: [{ color: '#131b26' }],
  },
  {
    featureType: 'poi.park',
    elementType: 'labels.text.fill',
    stylers: [{ color: '#38bdf8' }, { lightness: -20 }],
  },
  {
    featureType: 'road',
    elementType: 'geometry',
    stylers: [{ color: '#1e2433' }],
  },
  {
    featureType: 'road',
    elementType: 'geometry.stroke',
    stylers: [{ color: '#151923' }],
  },
  {
    featureType: 'road',
    elementType: 'labels.text.fill',
    stylers: [{ color: '#64748b' }],
  },
  {
    featureType: 'road.highway',
    elementType: 'geometry',
    stylers: [{ color: '#293247' }],
  },
  {
    featureType: 'road.highway',
    elementType: 'geometry.stroke',
    stylers: [{ color: '#181e2b' }],
  },
  {
    featureType: 'road.highway',
    elementType: 'labels.text.fill',
    stylers: [{ color: '#60a5fa' }],
  },
  {
    featureType: 'transit',
    elementType: 'geometry',
    stylers: [{ color: '#1b2130' }],
  },
  {
    featureType: 'transit.station',
    elementType: 'labels.text.fill',
    stylers: [{ color: '#93c5fd' }],
  },
  {
    featureType: 'water',
    elementType: 'geometry',
    stylers: [{ color: '#0a0c12' }],
  },
  {
    featureType: 'water',
    elementType: 'labels.text.fill',
    stylers: [{ color: '#475569' }],
  },
  {
    featureType: 'water',
    elementType: 'labels.text.stroke',
    stylers: [{ color: '#090b10' }],
  },
];
