import { useEffect, useState } from "react";

/** Views are hash routes so the FastAPI static mount needs no new backend routes. */
export const VIEWS = [
  { id: "weave", label: "The Weave", icon: "weave" },
  { id: "experiments", label: "Experiments", icon: "experiments" },
  { id: "institutions", label: "Institutions", icon: "institutions" },
  { id: "protection", label: "Protection", icon: "protection" },
  { id: "evidence", label: "Evidence", icon: "evidence" },
];

function parse(hash) {
  const [path, query = ""] = hash.replace(/^#\/?/, "").split("?");
  const view = VIEWS.some((v) => v.id === path) ? path : "weave";
  return { view, params: new URLSearchParams(query) };
}

export function useHashRoute() {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function routeHref(view, { simulated = false } = {}) {
  return `#/${view}${simulated ? "?sim=1" : ""}`;
}
