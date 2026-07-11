"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const POLL_INTERVAL_MS = 150;

export function usePngFrame(enabled: boolean) {
  const [src, setSrc] = useState<string | null>(null);
  const [available, setAvailable] = useState(false);
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let inFlight = false;

    const poll = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const response = await fetch(`/api/live-frame?ts=${Date.now()}`, { cache: "no-store" });
        if (!response.ok || cancelled) return;
        const nextUrl = URL.createObjectURL(await response.blob());
        if (cancelled) {
          URL.revokeObjectURL(nextUrl);
          return;
        }
        const previous = urlRef.current;
        urlRef.current = nextUrl;
        setSrc(nextUrl);
        setAvailable(true);
        if (previous) URL.revokeObjectURL(previous);
      } catch {
        // Preserve the last complete frame while the writer or route is unavailable.
      } finally {
        inFlight = false;
      }
    };

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [enabled]);

  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
  }, []);

  const reset = useCallback(() => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setSrc(null);
    setAvailable(false);
  }, []);

  return { src, available, reset };
}
