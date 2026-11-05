import { useEffect, useState } from "react"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { ScenariosView } from "@/components/scenarios-view"
import { MapApp } from "@/map-app"

type View = "map" | "scenarios"

const readView = (): View => (window.location.hash === "#scenarios" ? "scenarios" : "map")

const DOC_TITLE: Record<View, string> = {
  map: "Gauteng 2026 · Ward Predictions",
  scenarios: "Coalition Scenarios · Gauteng 2026",
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
    document.title = DOC_TITLE[view]
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

function ViewSwitch({ view, go }: { view: View; go: (v: View) => void }) {
  return (
    <nav aria-label="Views">
      <ToggleGroup type="single" variant="outline" size="sm" value={view} onValueChange={(v) => v && go(v as View)}>
        <ToggleGroupItem value="map" className="px-3">Ward map</ToggleGroupItem>
        <ToggleGroupItem value="scenarios" className="px-3">Coalition scenarios</ToggleGroupItem>
      </ToggleGroup>
    </nav>
  )
}

export default function App() {
  const [view, go] = useView()
  const nav = <ViewSwitch view={view} go={go} />

  if (view === "map") return <MapApp nav={nav} />

  return (
    <div className="min-h-svh">
      <header className="border-b bg-card/60 backdrop-blur">
        <div className="mx-auto flex max-w-[2560px] items-center justify-between gap-6 px-8 py-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Coalition scenarios</h1>
            <p className="text-sm text-muted-foreground">
              What could govern Gauteng's metros and province, read from coalition outcomes in other countries.
            </p>
          </div>
          {nav}
        </div>
      </header>
      <ScenariosView />
    </div>
  )
}
