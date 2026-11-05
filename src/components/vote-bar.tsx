import { cn } from "@/lib/utils"
import type { Role, Share } from "@/scenarios"

const ROLE_LABEL: Record<Role, string> = {
  gov: "in government",
  support: "outside support",
  out: "excluded",
}

// One neutral ramp: dark = governed, mid = kept it in office from outside,
// light = excluded. Remaining vote (small parties) is the empty track.
const ROLE_FILL: Record<Role, string> = {
  gov: "bg-foreground",
  support: "bg-foreground/45",
  out: "bg-muted-foreground/30",
}

function fmt(pct: number, approx?: boolean) {
  return approx ? `≈${Math.round(pct)}` : pct.toFixed(1)
}

export function RoleMark({ role }: { role: Role }) {
  return <span aria-hidden className={cn("inline-block size-2.5 shrink-0 rounded-[2px]", ROLE_FILL[role])} />
}

export function RoleKey({ className }: { className?: string }) {
  return (
    <div className={cn("flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground", className)}>
      {(["gov", "support", "out"] as Role[]).map((r) => (
        <span key={r} className="flex items-center gap-1.5">
          <RoleMark role={r} />
          {r === "gov" ? "In government" : r === "support" ? "Outside support" : "Excluded"}
        </span>
      ))}
      <span className="flex items-center gap-1.5">
        <span aria-hidden className="inline-block size-2.5 rounded-[2px] border bg-muted" />
        Smaller parties
      </span>
    </div>
  )
}

/** Vote shares as one 0–100% bar, coloured by each party's role in the government that formed. */
export function VoteBar({ shares, approx, compact }: { shares: Share[]; approx?: boolean; compact?: boolean }) {
  const label = shares.map((x) => `${x.party} ${fmt(x.pct, approx)}% ${ROLE_LABEL[x.role]}`).join(", ")
  return (
    <div className="grid gap-1.5">
      <div
        role="img"
        aria-label={`Vote shares: ${label}`}
        className={cn("flex w-full overflow-hidden rounded-[3px] bg-muted", compact ? "h-2" : "h-3")}
      >
        {shares.map((x) => (
          <span
            key={x.party}
            title={`${x.party} ${fmt(x.pct, approx)}%: ${ROLE_LABEL[x.role]}`}
            className={cn("h-full border-r-2 border-card last:border-r-0", ROLE_FILL[x.role])}
            style={{ width: `${x.pct}%` }}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
        {shares.map((x) => (
          <span key={x.party} className="inline-flex items-center gap-1">
            <RoleMark role={x.role} />
            <span className={x.role === "gov" ? "font-medium text-foreground" : "text-muted-foreground"}>{x.party}</span>
            <span className="tabular-nums text-muted-foreground">{fmt(x.pct, approx)}</span>
          </span>
        ))}
      </div>
    </div>
  )
}
