import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { StatusBadge } from "@/components/status-badge"
import { CountChart, ElectionsChart } from "@/components/ward-charts"
import { ShareBars, Verdict } from "@/components/ward-hover-card"
import type { WardSummary } from "@/lib/model"
import type { Election } from "@/types"

function Section({ title, description, children }: { title: string; description?: string; children: React.ReactNode }) {
  return (
    <section className="grid min-w-0 grid-cols-1 gap-3">
      <div>
        <h4 className="text-sm font-medium">{title}</h4>
        {description && <p className="text-xs text-muted-foreground">{description}</p>}
      </div>
      {children}
    </section>
  )
}

/** Everything about one ward: the current result and its vote share over time. */
export function WardDetail({
  ward,
  elections,
  order,
  onClose,
}: {
  ward: WardSummary
  elections: Election[]
  order: string[]
  onClose: () => void
}) {
  const { name, municipality, district } = ward.feature.properties
  const pastLabels = elections.filter((e) => ward.history?.[e.id]).map((e) => e.label)

  return (
    <div className="grid min-w-0 grid-cols-1 gap-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-lg leading-tight font-semibold">{name ?? ward.id}</h3>
          <p className="truncate text-sm text-muted-foreground">{[municipality, district].filter(Boolean).join(" · ")}</p>
          <p className="text-xs text-muted-foreground tabular-nums">Ward {ward.id}</p>
        </div>
        <div className="flex items-center gap-1">
          <StatusBadge status={ward.status} />
          <Button variant="ghost" size="icon" className="size-7" onClick={onClose} aria-label="Close ward">
            <X />
          </Button>
        </div>
      </div>

      <Section title="Current result">
        {ward.status !== "not_started" && <Verdict ward={ward} />}
        <ShareBars ward={ward} />
      </Section>

      <Separator />

      <Section
        title="Vote share across elections"
        description={
          pastLabels.length
            ? `${pastLabels.join(", ")} results, then the 2026 forecast and count`
            : "Past results, then the 2026 forecast and count"
        }
      >
        <ElectionsChart ward={ward} elections={elections} order={order} />
      </Section>

      <Separator />

      <Section title="Election night count" description="Running vote share at each results update">
        <CountChart ward={ward} order={order} />
      </Section>
    </div>
  )
}
