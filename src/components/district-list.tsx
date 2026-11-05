import { ranked } from "@/data"
import { SegmentedBar } from "@/components/segmented-bar"
import type { DistrictSummary } from "@/lib/model"
import { cn } from "@/lib/utils"
import { partyColor, partyName } from "@/parties"

/** One row per district: its prediction, the count so far, and progress. Click to focus it. */
export function DistrictList({
  districts,
  selected,
  order,
  onSelect,
  onHover,
}: {
  districts: DistrictSummary[]
  selected: string | null
  order: string[]
  onSelect: (name: string | null) => void
  onHover: (name: string | null) => void
}) {
  if (!districts.length) {
    return <p className="py-6 text-center text-sm text-muted-foreground">No districts in the ward data.</p>
  }
  return (
    <div className="grid gap-2" onMouseLeave={() => onHover(null)}>
      {districts.map((d) => {
        const [leader, share = 0] = ranked(d.predictedShares)[0] ?? []
        const holding = d.called ? Math.round((d.hits / d.called) * 100) : null
        const active = selected === d.name
        return (
          <button
            key={d.name}
            type="button"
            aria-pressed={active}
            onClick={() => onSelect(active ? null : d.name)}
            onMouseEnter={() => onHover(d.name)}
            onFocus={() => onHover(d.name)}
            onBlur={() => onHover(null)}
            className={cn(
              "grid gap-2.5 rounded-lg border p-3 text-left transition-colors outline-none hover:bg-accent/60 focus-visible:ring-[3px] focus-visible:ring-ring/50",
              active && "border-foreground/40 bg-accent/60",
            )}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate font-medium">{d.name}</div>
                <div className="text-xs text-muted-foreground">
                  {d.wardCount} wards · {d.municipalities.length} {d.municipalities.length === 1 ? "municipality" : "municipalities"}
                </div>
              </div>
              {leader && (
                <div className="text-right" title={`Predicted largest party: ${partyName(leader)}`}>
                  <div className="flex items-center justify-end gap-1.5 font-semibold">
                    <span className="size-2.5 rounded-[3px]" style={{ background: partyColor(leader) }} />
                    {leader} <span className="tabular-nums">{share.toFixed(1)}%</span>
                  </div>
                  <div className="text-[11px] text-muted-foreground">predicted</div>
                </div>
              )}
            </div>
            <div className="grid grid-cols-[3.25rem_1fr] items-center gap-x-2 gap-y-1.5 text-[11px] text-muted-foreground">
              <span>Forecast</span>
              <SegmentedBar variant="thin" shares={d.predictedShares} order={order} />
              <span>Counted</span>
              {d.votesCounted ? (
                <SegmentedBar variant="thin" shares={d.countedShares} order={order} />
              ) : (
                <span className="h-2 rounded-full bg-muted" />
              )}
            </div>
            <div className="flex gap-3 text-xs text-muted-foreground tabular-nums">
              <span>
                <span className="text-foreground">{d.reporting}</span>/{d.wardCount} reporting
              </span>
              <span>
                <span className="text-foreground">{holding == null ? "–" : `${holding}%`}</span> holding
              </span>
              <span>
                <span className="text-foreground">{d.upsets}</span> {d.upsets === 1 ? "upset" : "upsets"}
              </span>
            </div>
          </button>
        )
      })}
    </div>
  )
}
