import { useCallback, useMemo, useState } from "react"
import { Loader2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectSeparator, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { TooltipProvider } from "@/components/ui/tooltip"
import { DistrictList } from "@/components/district-list"
import { PartyChart } from "@/components/party-chart"
import { PredictionOverview } from "@/components/prediction-overview"
import { bucketShares } from "@/components/segmented-bar"
import { UpsetsTable } from "@/components/upsets-table"
import { WardDetail } from "@/components/ward-detail"
import { WardHoverCard } from "@/components/ward-hover-card"
import { WardMap, type HoverEvent, type WardFill } from "@/components/ward-map"
import { ranked } from "@/data"
import { useElectionData, type WardSummary } from "@/lib/model"
import { MAIN_PARTIES, OTHER, OTHER_COLOR, partyColor, partyName } from "@/parties"

type ColourBy = "prediction" | "result"
type Panel = "districts" | "parties" | "upsets" | "ward"

const ALL = "__all"
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
      <span className="flex items-center gap-1.5">
        <svg width="18" height="10" aria-hidden>
          <line x1="0" y1="5" x2="18" y2="5" stroke="var(--district-boundary)" strokeWidth="2.25" strokeLinecap="round" />
        </svg>
        District border
      </span>
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

export default function App() {
  const { state, summaries, overall, districts, elections } = useElectionData()
  const [colourBy, setColourBy] = useState<ColourBy>("prediction")
  const [district, setDistrict] = useState<string | null>(null)
  const [hoverDistrict, setHoverDistrict] = useState<string | null>(null)
  const [selectedWardId, setSelectedWardId] = useState<string | null>(null)
  const [panel, setPanel] = useState<Panel>("districts")
  const [hover, setHover] = useState<HoverEvent | null>(null)
  const [tableHoverId, setTableHoverId] = useState<string | null>(null)

  const byId = useMemo(() => new Map(summaries.map((s) => [s.id, s])), [summaries])
  const fill = useCallback((id: string) => fillFor(byId.get(id), colourBy), [byId, colourBy])
  const hovered = hover ? byId.get(hover.id) : undefined
  const selectedWard = selectedWardId ? byId.get(selectedWardId) : undefined
  const focused = district ? districts.find((d) => d.name === district) : undefined

  // One block order for every segmented bar and line chart: overall forecast, largest first, "Other" last.
  const order = useMemo(() => {
    const keys = ranked(bucketShares(overall.predictedShares)).map(([p]) => p).filter((p) => p !== OTHER)
    return [...keys, OTHER]
  }, [overall])

  const selectDistrict = useCallback(
    (name: string | null) => {
      setDistrict(name)
      // Close the ward panel if that ward isn't in the newly chosen district.
      if (name && selectedWardId && byId.get(selectedWardId)?.district !== name) {
        setSelectedWardId(null)
        setPanel((p) => (p === "ward" ? "districts" : p))
      }
    },
    [selectedWardId, byId],
  )

  const selectWard = useCallback(
    (id: string) => {
      setSelectedWardId(id)
      setPanel("ward")
      const d = byId.get(id)?.district
      if (d) setDistrict(d)
    },
    [byId],
  )

  const closeWard = useCallback(() => {
    setSelectedWardId(null)
    setPanel("districts")
  }, [])

  if (state.status === "loading") {
    return (
      <div className="grid min-h-svh place-items-center text-muted-foreground">
        <Loader2 className="size-6 animate-spin" />
      </div>
    )
  }
  if (state.status === "error") {
    return <div className="grid min-h-svh place-items-center p-6 text-destructive">{state.message}</div>
  }

  const { predictions, results, wards } = state
  const updated = results.updated_at ? new Date(results.updated_at) : null
  const isSample = predictions.sample || results.sample
  const scopeWards = focused ? focused.wards : summaries
  const scope = focused ?? overall

  return (
    <TooltipProvider>
      <div className="min-h-svh">
        <header className="border-b bg-card/60 backdrop-blur">
          <div className="mx-auto flex max-w-[1400px] items-end justify-between gap-6 px-6 py-5">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">Election predictions</h1>
              <p className="text-sm text-muted-foreground">
                Forecast and live count, from the whole area down to each district and ward.
              </p>
            </div>
            <div className="flex items-center gap-3 text-sm text-muted-foreground">
              {isSample && (
                <Badge variant="outline" className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300">
                  Sample data
                </Badge>
              )}
              {updated && (
                <span>
                  Results updated{" "}
                  {updated.toLocaleString("en-ZA", { dateStyle: "medium", timeStyle: "short", timeZone: "Africa/Johannesburg" })}{" "}
                  SAST
                </span>
              )}
              {districts.length > 0 && (
                <Select value={district ?? ALL} onValueChange={(v) => selectDistrict(v === ALL ? null : v)}>
                  <SelectTrigger className="w-52 bg-card" aria-label="District">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent align="end">
                    <SelectItem value={ALL}>All districts</SelectItem>
                    <SelectSeparator />
                    {districts.map((d) => (
                      <SelectItem key={d.name} value={d.name}>
                        {d.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>
          </div>
        </header>

        <main className="mx-auto grid max-w-[1400px] gap-4 px-6 py-6">
          {isSample && (
            <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-sm text-amber-800 dark:text-amber-200">
              Showing synthetic sample data. Ward shapes, districts, predictions, past results and the count are all
              made up. See the README to load real data.
            </p>
          )}

          <PredictionOverview
            scope={scope}
            order={order}
            title={focused ? `${focused.name} prediction` : "General prediction"}
            description={
              focused
                ? `${focused.wardCount} wards · ${focused.municipalities.join(", ")}`
                : `All ${overall.wardCount} wards${districts.length ? ` across ${districts.length} districts` : ""}`
            }
          />

          <div className="grid grid-cols-[minmax(0,1fr)_400px] gap-4">
            <Card className="min-w-0 gap-4">
              <CardHeader>
                <CardTitle>Ward map</CardTitle>
                <CardDescription>
                  {colourBy === "prediction" ? "Coloured by predicted winner" : "Coloured by current leader"} · scroll
                  to zoom · click a ward for its history
                </CardDescription>
                <CardAction>
                  <ToggleGroup
                    type="single"
                    variant="outline"
                    size="sm"
                    value={colourBy}
                    onValueChange={(v) => v && setColourBy(v as ColourBy)}
                  >
                    <ToggleGroupItem value="prediction" className="px-3">
                      Prediction
                    </ToggleGroupItem>
                    <ToggleGroupItem value="result" className="px-3">
                      Current result
                    </ToggleGroupItem>
                  </ToggleGroup>
                </CardAction>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="h-[min(72vh,680px)] min-h-[440px] overflow-hidden rounded-lg border bg-muted/30">
                  <WardMap
                    wards={wards}
                    fillFor={fill}
                    highlightedId={hover?.id ?? tableHoverId}
                    selectedId={selectedWardId}
                    onHover={setHover}
                    onSelectWard={selectWard}
                    focusDistrict={district}
                    highlightDistrict={hoverDistrict}
                    onSelectDistrict={selectDistrict}
                  />
                </div>
                <MapLegend by={colourBy} />
              </CardContent>
            </Card>

            <Card className="min-w-0 gap-4">
              <Tabs value={panel} onValueChange={(v) => setPanel(v as Panel)} className="gap-4">
                <CardHeader>
                  <TabsList className="w-full">
                    <TabsTrigger value="districts">Districts</TabsTrigger>
                    <TabsTrigger value="parties">Parties</TabsTrigger>
                    <TabsTrigger value="upsets">
                      Upsets
                      <Badge variant="secondary" className="h-4.5 min-w-4.5 px-1 tabular-nums">
                        {scope.upsets}
                      </Badge>
                    </TabsTrigger>
                    <TabsTrigger value="ward" disabled={!selectedWard}>
                      Ward
                    </TabsTrigger>
                  </TabsList>
                </CardHeader>
                <CardContent>
                  <TabsContent value="districts" className="grid gap-3">
                    <p className="text-sm text-muted-foreground">
                      Forecast vs count by district. Click one to zoom in; hover to find it on the map.
                    </p>
                    <DistrictList
                      districts={districts}
                      selected={district}
                      order={order}
                      onSelect={selectDistrict}
                      onHover={setHoverDistrict}
                    />
                  </TabsContent>
                  <TabsContent value="parties" className="grid gap-3">
                    <p className="text-sm text-muted-foreground">
                      Wards each party is predicted to win vs wards it leads now
                      {focused ? ` in ${focused.name}` : ""}.
                    </p>
                    <PartyChart wards={scopeWards} />
                  </TabsContent>
                  <TabsContent value="upsets" className="grid gap-3">
                    <p className="text-sm text-muted-foreground">
                      Wards where someone other than the predicted winner leads{focused ? ` in ${focused.name}` : ""}.
                      Click one for its detail.
                    </p>
                    <div className="max-h-[560px] overflow-auto">
                      <UpsetsTable wards={scopeWards} onHoverWard={setTableHoverId} onSelectWard={selectWard} />
                    </div>
                  </TabsContent>
                  <TabsContent value="ward">
                    {selectedWard ? (
                      <WardDetail ward={selectedWard} elections={elections} order={order} onClose={closeWard} />
                    ) : (
                      <p className="py-6 text-center text-sm text-muted-foreground">Click a ward on the map to see its detail.</p>
                    )}
                  </TabsContent>
                </CardContent>
              </Tabs>
            </Card>
          </div>
        </main>

        {hover && hovered && <WardHoverCard ward={hovered} x={hover.x} y={hover.y} />}
      </div>
    </TooltipProvider>
  )
}
