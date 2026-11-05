import { useCallback, useMemo, useState } from "react"
import { Loader2 } from "lucide-react"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { PartyChart } from "@/components/party-chart"
import { StatCards } from "@/components/stat-cards"
import { UpsetsTable } from "@/components/upsets-table"
import { WardHoverCard } from "@/components/ward-hover-card"
import { WardMap, type HoverEvent, type WardFill } from "@/components/ward-map"
import type { useElectionData, WardSummary } from "@/lib/model"
import { MAIN_PARTIES, OTHER_COLOR, partyColor, partyName } from "@/parties"

type ColourBy = "prediction" | "result"

const NO_DATA: WardFill = { color: "var(--muted)", opacity: 1 }

function fillFor(w: WardSummary | undefined, by: ColourBy): WardFill {
  if (!w) return NO_DATA
  if (by === "prediction") {
    if (!w.predictedWinner) return NO_DATA
    const conf = Math.max(0, Math.min(1, w.confidence ?? 0.8))
    return { color: partyColor(w.predictedWinner), opacity: 0.3 + 0.7 * conf }
  }
  if (!w.leader) return NO_DATA
  return { color: partyColor(w.leader), opacity: w.status === "final" ? 1 : 0.5 }
}

function MapLegend({ by }: { by: ColourBy }) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-muted-foreground">
      {MAIN_PARTIES.map((p) => (
        <span key={p} className="flex items-center gap-1.5" title={partyName(p)}>
          <span className="size-2.5 rounded-[3px]" style={{ background: partyColor(p) }} />
          <span className="text-foreground">{p}</span>
        </span>
      ))}
      <span className="flex items-center gap-1.5">
        <span className="size-2.5 rounded-[3px]" style={{ background: OTHER_COLOR }} />
        Other
      </span>
      {by === "result" && (
        <span className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-[3px] border bg-muted" />
          No results yet
        </span>
      )}
      <span className="ml-auto flex items-center gap-2">
        {by === "prediction" ? "Less → more confident" : "Counting → final"}
        <span className="flex h-2.5 w-16 overflow-hidden rounded-[3px]">
          {(by === "prediction" ? [0.35, 0.6, 0.8, 1] : [0.5, 1]).map((o) => (
            <span key={o} className="flex-1 bg-foreground" style={{ opacity: o }} />
          ))}
        </span>
      </span>
    </div>
  )
}

export function MapView({ data }: { data: ReturnType<typeof useElectionData> }) {
  const { state, summaries } = data
  const [colourBy, setColourBy] = useState<ColourBy>("prediction")
  const [hover, setHover] = useState<HoverEvent | null>(null)
  const [tableHoverId, setTableHoverId] = useState<string | null>(null)

  const byId = useMemo(() => new Map(summaries.map((s) => [s.id, s])), [summaries])
  const fill = useCallback((id: string) => fillFor(byId.get(id), colourBy), [byId, colourBy])
  const hovered = hover ? byId.get(hover.id) : undefined

  if (state.status === "loading") {
    return (
      <div className="grid min-h-[60vh] place-items-center text-muted-foreground">
        <Loader2 className="size-6 animate-spin" />
      </div>
    )
  }
  if (state.status === "error") {
    return <div className="grid min-h-[60vh] place-items-center p-6 text-destructive">{state.message}</div>
  }

  const { predictions, results, wards } = state
  const isSample = predictions.sample || results.sample

  return (
    <>
      <main className="mx-auto grid max-w-[1400px] gap-4 px-4 py-6 sm:px-6">
        {isSample && (
          <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-800 dark:text-amber-200">
            Showing synthetic sample data. Ward shapes, predictions and results are made up. See the README to load real data.
          </p>
        )}

        <StatCards wards={summaries} />

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
          <Card className="min-w-0 gap-4">
            <CardHeader className="max-sm:has-data-[slot=card-action]:grid-cols-1">
              <CardTitle>Ward map</CardTitle>
              <CardDescription>
                {colourBy === "prediction" ? "Coloured by predicted winner" : "Coloured by current leader"}
                <span className="hidden sm:inline"> · scroll to zoom, drag to pan</span>
              </CardDescription>
              <CardAction className="max-sm:col-start-1 max-sm:row-start-3 max-sm:mt-2 max-sm:justify-self-start">
                <ToggleGroup
                  type="single"
                  variant="outline"
                  size="sm"
                  value={colourBy}
                  onValueChange={(v) => v && setColourBy(v as ColourBy)}
                >
                  <ToggleGroupItem value="prediction" className="px-3">Prediction</ToggleGroupItem>
                  <ToggleGroupItem value="result" className="px-3">Current result</ToggleGroupItem>
                </ToggleGroup>
              </CardAction>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="aspect-square overflow-hidden rounded-lg border bg-muted/30 sm:aspect-auto sm:h-[min(68vh,640px)] sm:min-h-[360px]">
                <WardMap
                  wards={wards}
                  fillFor={fill}
                  highlightedId={hover?.id ?? tableHoverId}
                  onHover={setHover}
                />
              </div>
              <MapLegend by={colourBy} />
            </CardContent>
          </Card>

          <div className="grid min-w-0 grid-cols-1 content-start gap-4">
            <Card className="gap-4">
              <CardHeader>
                <CardTitle>Wards by party</CardTitle>
                <CardDescription>Predicted wins vs wards each party leads now</CardDescription>
              </CardHeader>
              <CardContent>
                <PartyChart wards={summaries} />
              </CardContent>
            </Card>

            <Card className="gap-2">
              <CardHeader>
                <CardTitle>Upsets</CardTitle>
                <CardDescription>Hover a row to find it on the map</CardDescription>
              </CardHeader>
              <CardContent className="max-h-80 overflow-auto">
                <UpsetsTable wards={summaries} onHoverWard={setTableHoverId} />
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {hover && hovered && <WardHoverCard ward={hovered} x={hover.x} y={hover.y} />}
    </>
  )
}
