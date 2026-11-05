import { useEffect, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { MapView } from "@/components/map-view"
import { ScenariosView } from "@/components/scenarios-view"
import { useElectionData } from "@/lib/model"

type View = "map" | "scenarios"

const readView = (): View => (window.location.hash === "#scenarios" ? "scenarios" : "map")

const COPY: Record<View, { title: string; subtitle: string; docTitle: string }> = {
  map: {
    title: "Ward predictions",
    subtitle: "Every ward coloured by its forecast winner. Hover a ward for the current result.",
    docTitle: "Ward Predictions",
  },
  scenarios: {
    title: "Coalition scenarios",
    subtitle: "What could govern Gauteng's metros and province, read from coalition outcomes in other countries.",
    docTitle: "Coalition Scenarios · Ward Predictions",
  },
}

/** Current view, kept in the URL hash (#scenarios) so it can be linked and survives reloads. */
function useView() {
  const [view, setView] = useState<View>(readView)

  useEffect(() => {
    const sync = () => setView(readView())
    window.addEventListener("hashchange", sync)
    window.addEventListener("popstate", sync)
    return () => {
      window.removeEventListener("hashchange", sync)
      window.removeEventListener("popstate", sync)
    }
  }, [])

  useEffect(() => {
    document.title = COPY[view].docTitle
  }, [view])

  const go = (next: View) => {
    if (next === view) return
    const url = next === "scenarios" ? "#scenarios" : window.location.pathname + window.location.search
    window.history.pushState(null, "", url)
    setView(next)
    window.scrollTo({ top: 0 })
  }

  return [view, go] as const
}

export default function App() {
  const [view, go] = useView()
  // Load (and keep polling) results regardless of view, so switching back to the map is instant.
  const data = useElectionData()
  const { state } = data

  const ready = state.status === "ready" ? state : null
  const updated = ready?.results.updated_at ? new Date(ready.results.updated_at) : null
  const isSample = ready && (ready.predictions.sample || ready.results.sample)

  return (
    <div className="min-h-svh">
      <header className="border-b bg-card/60 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-end justify-between gap-3 px-4 py-5 sm:px-6">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">{COPY[view].title}</h1>
            <p className="text-sm text-muted-foreground">{COPY[view].subtitle}</p>
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-sm text-muted-foreground">
            {view === "map" && isSample && (
              <Badge variant="outline" className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300">
                Sample data
              </Badge>
            )}
            {view === "map" && updated && (
              <span>
                Results updated{" "}
                {updated.toLocaleString("en-ZA", { dateStyle: "medium", timeStyle: "short" })}
              </span>
            )}
            <nav aria-label="Views">
              <ToggleGroup type="single" variant="outline" size="sm" value={view} onValueChange={(v) => v && go(v as View)}>
                <ToggleGroupItem value="map" className="px-3">Ward map</ToggleGroupItem>
                <ToggleGroupItem value="scenarios" className="px-3">Coalition scenarios</ToggleGroupItem>
              </ToggleGroup>
            </nav>
          </div>
        </div>
      </header>

      {view === "scenarios" ? <ScenariosView /> : <MapView data={data} />}
    </div>
  )
}
