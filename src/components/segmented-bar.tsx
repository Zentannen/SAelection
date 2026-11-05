import { useEffect, useRef, useState } from "react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { colourBucket, partyColor, partyInk, partyName } from "@/parties"
import type { Shares } from "@/types"

/** Fold parties without their own colour into "Other" so no two blocks share a colour. */
export function bucketShares(shares: Shares): Shares {
  const out: Shares = {}
  for (const [p, s] of Object.entries(shares)) {
    const key = colourBucket(p)
    out[key] = (out[key] ?? 0) + s
  }
  return out
}

interface Props {
  shares: Shares
  /** Fixed block order (party codes, already bucketed) so bars line up across the page. */
  order: string[]
  variant?: "tall" | "thin"
  className?: string
  "aria-label"?: string
}

/**
 * One whole split into parts. Tall: 28px blocks with the share printed inside
 * when it fits. Thin: 8px with no text. Every block has a hover tooltip.
 */
export function SegmentedBar({ shares, order, variant = "tall", className, ...rest }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el || variant !== "tall") return
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [variant])

  const bucketed = bucketShares(shares)
  const keys = [...order, ...Object.keys(bucketed).filter((k) => !order.includes(k))]
  const total = keys.reduce((sum, k) => sum + (bucketed[k] ?? 0), 0)
  const parts = keys
    .map((party) => ({ party, value: total ? ((bucketed[party] ?? 0) / total) * 100 : 0 }))
    .filter((p) => p.value > 0)
  const tall = variant === "tall"

  return (
    <div
      ref={ref}
      role="img"
      aria-label={rest["aria-label"] ?? parts.map((p) => `${partyName(p.party)} ${p.value.toFixed(1)}%`).join(", ")}
      className={cn("flex w-full gap-[2px]", tall ? "h-7" : "h-2 overflow-hidden rounded-full", className)}
    >
      {parts.map(({ party, value }) => {
        // One decimal when it fits (so 21.5% and 21.2% read differently), else whole number, else nothing.
        const px = (value / 100) * width
        const room = (text: string) => text.length * 7 + 16
        const precise = `${value.toFixed(1)}%`
        const rounded = `${Math.round(value)}%`
        const label = !tall ? "" : px >= room(precise) ? precise : px >= room(rounded) ? rounded : ""
        return (
          <Tooltip key={party}>
            <TooltipTrigger asChild>
              <div
                className={cn(
                  "flex min-w-0 items-center transition-[flex-grow] duration-500",
                  tall ? "rounded-[4px] px-2 text-[11px] font-semibold tabular-nums" : "rounded-[1px]",
                )}
                style={{ flexGrow: value, flexBasis: 0, background: partyColor(party), color: partyInk(party) }}
              >
                {label}
              </div>
            </TooltipTrigger>
            <TooltipContent side="top" className="tabular-nums">
              {party} · {partyName(party)} — {value.toFixed(1)}%
            </TooltipContent>
          </Tooltip>
        )
      })}
    </div>
  )
}

/** Legend for segmented bars: names in block order, no numbers. */
export function SegmentLegend({ order, className }: { order: string[]; className?: string }) {
  return (
    <div className={cn("flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground", className)}>
      {order.map((p) => (
        <span key={p} className="flex items-center gap-1.5" title={partyName(p)}>
          <span className="size-2.5 rounded-[3px]" style={{ background: partyColor(p) }} />
          <span className="text-foreground">{p}</span>
        </span>
      ))}
    </div>
  )
}
