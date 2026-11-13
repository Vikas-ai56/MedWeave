import { useEffect, useRef, useState } from "react";

export function useNow(intervalMs) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
  return now;
}

export function usePrefersReducedMotion() {
  const query = "(prefers-reduced-motion: reduce)";
  const [reduced, setReduced] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const media = window.matchMedia(query);
    const onChange = () => setReduced(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

export function useElementSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (!ref.current) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize((prev) => (prev.width === width && prev.height === height ? prev : { width, height }));
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

/** Client time of the last successful response; each good poll yields a new object. */
export function useUpdatedAt(data) {
  const [updatedAt, setUpdatedAt] = useState(null);
  useEffect(() => {
    if (data) setUpdatedAt(Date.now());
  }, [data]);
  return updatedAt;
}

/**
 * Counts completed turns and reports when a new one lands, so The Weave can play its
 * round transition. The first value seen never triggers it.
 */
export function useTurnCompletion(completedTurns, sessionId) {
  const previous = useRef({ sessionId, completedTurns });
  const [pulse, setPulse] = useState(0);
  useEffect(() => {
    const before = previous.current;
    if (before.sessionId === sessionId && completedTurns > before.completedTurns) setPulse((n) => n + 1);
    previous.current = { sessionId, completedTurns };
  }, [completedTurns, sessionId]);
  return pulse;
}

export function useEscape(onEscape, active) {
  useEffect(() => {
    if (!active) return undefined;
    const onKey = (event) => event.key === "Escape" && onEscape();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onEscape, active]);
}

export function useCopy() {
  const [copied, setCopied] = useState(null);
  useEffect(() => {
    if (!copied) return undefined;
    const timer = setTimeout(() => setCopied(null), 1600);
    return () => clearTimeout(timer);
  }, [copied]);
  const copy = (text) => {
    navigator.clipboard?.writeText(text).then(() => setCopied(text), () => setCopied(null));
  };
  return { copied, copy };
}
