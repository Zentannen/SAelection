import { useLayoutEffect, useRef, useState } from "react"
import { CircleCheck, CircleDashed, TriangleAlert } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { partyColor, partyName } from "@/parties"
import type { WardSummary } from "@/lib/model"
import { StatusBadge } from "@/components/status-badge"

const pct = (n: number) => `${n.toFixed(1)}%`
const int = (n: number) => n.toLocaleString("en-ZA")

/** Party rows: solid bar = current share, tick = predicted share. */
export function ShareBars({ ward, limit = 5 }: { ward: WardSummary; limit?: number }) {
  const rows = ward.rows.slice(0, limit)
  const hasResult = ward.status !== "not_started"
  const max = Math.max(...rows.map((r) => Math.max(r.current ?? 0, r.predicted ?? 0)), 1)
  const scale = (v: number) => `${(v / Math.max(max * 1.08, 50)) * 100}%`

  return (
    <div className="grid min-w-0 grid-cols-1 gap-2">
      {rows.map((r) => (
        <div key={r.party} className="grid grid-cols-[4.5rem_1fr_3rem] items-center gap-2.5 text-xs">
          <span className="flex min-w-0 items-center gap-1.5" title={partyName(r.party)}>
            <span className="size-2 shrink-0 rounded-[2px]" style={{ background: partyColor(r.party) }} />
            <span className="truncate font-medium">{r.party}</span>
          </span>
          <span className="relative h-2 rounded-full bg-muted">
            {hasResult && r.current != null && (
              <span
                className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-500"
                style={{ width: scale(r.current), background: partyColor(r.party) }}
              />
            )}
            {!hasResult && r.predicted != null && (
              <span
                className="absolute inset-y-0 left-0 rounded-full opacity-35"
                style={{ width: scale(r.predicted), background: partyColor(r.party) }}
              />
            )}
            {hasResult && r.predicted != null && (
              <span
                className="absolute -top-1 -bottom-1 w-0.5 -translate-x-1/2 rounded-full bg-foreground ring-2 ring-popover"
                style={{ left: scale(r.predicted) }}
                title={`Predicted ${pct(r.predicted)}`}
              />
            )}
          </span>
          <span className="text-right tabular-nums">
            {hasResult ? pct(r.current ?? 0) : r.predicted != null ? pct(r.predicted) : "–"}
          </span>
        </div>
      ))}
      <div className="mt-0.5 flex items-center gap-4 text-[11px] text-muted-foreground">
        {hasResult ? (
          <>
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-4 rounded-full bg-muted-foreground/60" /> Current
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-3 w-0.5 rounded-full bg-foreground" /> Predicted
            </span>
          </>
        ) : (
          <span>Predicted shares (no votes counted yet)</span>
        )}
      </div>
    </div>
  )
}

export function Verdict({ ward }: { ward: WardSummary }) {
  if (!ward.leader || !ward.predictedWinner) return null
  const done = ward.status === "final"
  if (!ward.upset) {
    return (
      <p className="flex items-center gap-1.5 text-sm font-medium">
        <CircleCheck className="size-4 text-good" />
        {ward.leader} {done ? "won" : "leads"} as predicted
        <span className="font-normal text-muted-foreground">+{ward.leadMargin?.toFixed(1)} pts</span>
      </p>
    )
  }
  return (
    <p className="flex items-center gap-1.5 text-sm font-medium">
      <TriangleAlert className="size-4 text-critical" />
      {ward.leader} {done ? "won" : "leads"}, predicted {ward.predictedWinner}
    </p>
  )
}

export function WardHoverCard({ ward, x, y }: { ward: WardSummary; x: number; y: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ left: x, top: y })

  // Keep the card on-screen: flip left/up near the viewport edges.
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const gap = 16
    const { offsetWidth: w, offsetHeight: h } = el
    let left = x + gap
    let top = y + gap
    if (left + w > window.innerWidth - 8) left = x - w - gap
    if (top + h > window.innerHeight - 8) top = Math.max(8, window.innerHeight - h - 8)
    setPos({ left: Math.max(8, left), top })
  }, [x, y, ward])

  const { name, municipality, district } = ward.feature.properties
  const turnout = ward.registered && ward.votesCounted ? (ward.votesCounted / ward.registered) * 100 : undefined

  return (
    <div
      ref={ref}
      role="tooltip"
      className="pointer-events-none fixed z-50 w-80 max-w-[calc(100vw-16px)] animate-in rounded-xl border bg-popover p-4 text-popover-foreground shadow-xl fade-in-0 zoom-in-95"
      style={pos}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate leading-tight font-semibold">{name ?? ward.id}</h3>
          <p className="truncate text-xs text-muted-foreground">
            {[municipality, district].filter(Boolean).join(" · ") || ward.id}
          </p>
        </div>
        <StatusBadge status={ward.status} />
      </div>

      <Separator className="my-3" />

      <div className="grid gap-3">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Current result</span>
          {ward.status !== "not_started" && (
            <span className="text-xs text-muted-foreground tabular-nums">
              {int(ward.votesCounted)} votes{turnout != null && ` · ${turnout.toFixed(0)}% turnout`}
            </span>
          )}
        </div>
        {ward.status === "not_started" ? (
          <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
            <CircleDashed className="size-4" /> No votes counted yet
          </p>
        ) : (
          <Verdict ward={ward} />
        )}
        <ShareBars ward={ward} />
      </div>

      {ward.predictedWinner && (
        <>
          <Separator className="my-3" />
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Predicted winner</span>
            <span className="flex items-center gap-2">
              <Badge variant="outline" className="gap-1.5">
                <span className="size-2 rounded-[2px]" style={{ background: partyColor(ward.predictedWinner) }} />
                {ward.predictedWinner}
              </Badge>
              {ward.confidence != null && (
                <span className="text-muted-foreground tabular-nums">{Math.round(ward.confidence * 100)}% conf.</span>
              )}
            </span>
          </div>
        </>
      )}
      <p className="mt-3 text-[11px] text-muted-foreground">Click for vote share over time</p>
    </div>
  )
}
