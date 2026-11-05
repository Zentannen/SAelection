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
import { cn } from "@/lib/utils"
import { MAIN_PARTIES, OTHER, OTHER_COLOR, colourBucket, partyColor, partyName } from "@/parties"

type ColourBy = "prediction" | "result"
type Panel = "districts" | "parties" | "upsets" | "ward"

const ALL = "__all"
const NO_DATA: WardFill = { color: "var(--muted)", opacity: 1 }

function fillFor(w: WardSummary | undefined, by: ColourBy, party: string | null): WardFill {
  if (!w) return NO_DATA
  const winner = by === "prediction" ? w.predictedWinner : w.leader
  if (!winner) return NO_DATA
  // A highlighted party: its wards keep their colour, every other ward goes grey.
  if (party && colourBucket(winner) !== party) return NO_DATA
  if (by === "prediction") {
    const conf = Math.max(0, Math.min(1, w.confidence ?? 0.8))
    return { color: partyColor(winner), opacity: 0.3 + 0.7 * conf }
  }
  return { color: partyColor(winner), opacity: w.status === "final" ? 1 : 0.5 }
}

function MapLegend({
  by,
  active,
  pinned,
  onHover,
  onToggle,
}: {
  by: ColourBy
  active: string | null
  pinned: string | null
  onHover: (party: string | null) => void
  onToggle: (party: string) => void
}) {
  return (
    <div className="grid gap-2.5">
      <div className="flex flex-wrap items-center gap-1.5" onMouseLeave={() => onHover(null)}>
        {[...MAIN_PARTIES, OTHER].map((p) => (
          <button
            key={p}
            type="button"
            aria-pressed={pinned === p}
            title={`${partyName(p)}: ${pinned === p ? "click to clear" : "hover or click to highlight its wards"}`}
            onMouseEnter={() => onHover(p)}
            onFocus={() => onHover(p)}
            onBlur={() => onHover(null)}
            onClick={() => onToggle(p)}
            className={cn(
              "flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs font-medium transition-[opacity,background-color] outline-none hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/50",
              pinned === p && "border-foreground/50 bg-accent",
              active && active !== p && "opacity-40",
            )}
          >
            <span className="size-2.5 rounded-[3px]" style={{ background: p === OTHER ? OTHER_COLOR : partyColor(p) }} />
            {p}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-muted-foreground">
        <span>{pinned ? `Showing ${partyName(pinned)} wards · click again to clear` : "Hover or click a party to highlight its wards"}</span>
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
  const [hoverParty, setHoverParty] = useState<string | null>(null)
  const [pinnedParty, setPinnedParty] = useState<string | null>(null)
  const activeParty = hoverParty ?? pinnedParty

  const byId = useMemo(() => new Map(summaries.map((s) => [s.id, s])), [summaries])
  const fill = useCallback((id: string) => fillFor(byId.get(id), colourBy, activeParty), [byId, colourBy, activeParty])
  const hovered = hover ? byId.get(hover.id) : undefined
  const selectedWard = selectedWardId ? byId.get(selectedWardId) : undefined
  const focused = district ? districts.find((d) => d.name === district) : undefined

  // One block order for every segmented bar and line chart: overall forecast, largest first, "Other" last.
  // Every coloured party is always included (after the forecast-ranked ones), so a
  // party the forecast leaves out still gets its own colour, legend entry and line.
  const order = useMemo(() => {
    const keys = ranked(bucketShares(overall.predictedShares)).map(([p]) => p).filter((p) => p !== OTHER)
    return [...keys, ...MAIN_PARTIES.filter((p) => !keys.includes(p)), OTHER]
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
      // Leaving the list/table that triggered this unmounts it, so clear its hover outline.
      setTableHoverId(null)
      setHoverDistrict(null)
      // Follow the ward's district; a ward with no district shows the whole area.
      setDistrict(byId.get(id)?.district ?? null)
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
      {/* Desktop: the page fills the window and each column scrolls on its own. */}
      <div className="flex min-h-svh flex-col desk:h-svh">
        <header className="shrink-0 border-b bg-card/60 backdrop-blur">
          <div className="mx-auto flex max-w-[2560px] items-center justify-between gap-6 px-8 py-4">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">Election predictions</h1>
              <p className="text-sm text-muted-foreground">
                Forecast and live count, from the whole area down to each district and ward.
                {isSample && (
                  <span className="ml-2 text-amber-700 dark:text-amber-300">
                    Synthetic sample data: wards, districts, forecasts, past results and the count are all made up.
                  </span>
                )}
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

        <main className="mx-auto grid w-full max-w-[2560px] flex-1 grid-cols-[minmax(0,1fr)_420px] gap-5 px-8 py-6 desk:min-h-0 desk:grid-cols-[400px_minmax(0,1fr)_440px] desk:grid-rows-[minmax(0,1fr)]">
          <div className="col-span-2 grid min-w-0 desk:col-span-1 desk:min-h-0">
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
          </div>

            <Card className="min-w-0 gap-4 desk:min-h-0">
              <CardHeader>
                <CardTitle>Ward map</CardTitle>
                <CardDescription>
                  {colourBy === "prediction" ? "Coloured by predicted winner" : "Coloured by current leader"} ·{" "}
                  {/* In the full-height desktop layout the page doesn't scroll, so plain scrolling zooms the map. */}
                  <span className="hidden desk:inline">scroll to zoom</span>
                  <span className="desk:hidden">Ctrl + scroll to zoom</span> · click a ward for its history
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
              <CardContent className="flex flex-col gap-4 desk:min-h-0 desk:flex-1">
                <div className="h-[min(72vh,760px)] min-h-[480px] overflow-hidden rounded-lg border bg-muted/30 desk:h-auto desk:min-h-0 desk:flex-1">
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
                <MapLegend
                  by={colourBy}
                  active={activeParty}
                  pinned={pinnedParty}
                  onHover={setHoverParty}
                  onToggle={(p) => setPinnedParty((cur) => (cur === p ? null : p))}
                />
              </CardContent>
            </Card>

            <Card className="min-w-0 gap-4 desk:min-h-0">
              <Tabs
                value={panel}
                onValueChange={(v) => {
                  setPanel(v as Panel)
                  setTableHoverId(null)
                  setHoverDistrict(null)
                }}
                className="gap-4 desk:min-h-0 desk:flex-1"
              >
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
                <CardContent className="desk:min-h-0 desk:flex-1 desk:overflow-auto">
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
                    <div className="max-h-[640px] overflow-auto desk:max-h-none">
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
        </main>

        {hover && hovered && <WardHoverCard ward={hovered} x={hover.x} y={hover.y} />}
      </div>
    </TooltipProvider>
  )
}
