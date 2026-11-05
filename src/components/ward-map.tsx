import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { geoMercator, geoPath } from "d3-geo"
import { select } from "d3-selection"
import "d3-transition"
import { zoom, zoomIdentity, type ZoomBehavior } from "d3-zoom"
import { Maximize2, Minus, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { WardCollection } from "@/types"

export interface WardFill {
  color: string
  opacity: number
}

export interface HoverEvent {
  id: string
  x: number
  y: number
}

interface Props {
  wards: WardCollection
  fillFor: (id: string) => WardFill
  highlightedId?: string | null
  onHover: (e: HoverEvent | null) => void
}

const PAD = 12
const MAX_ZOOM = 16

export function WardMap({ wards, fillFor, highlightedId, onHover }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const layerRef = useRef<SVGGElement>(null)
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [scale, setScale] = useState(1)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize({ width, height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Pan & zoom: wheel / pinch / drag. The transform is applied straight to the
  // <g> (no React re-render per frame), so it stays smooth with thousands of wards.
  useEffect(() => {
    const svg = svgRef.current
    if (!svg || !size.width) return
    const behaviour = zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, MAX_ZOOM])
      .translateExtent([
        [0, 0],
        [size.width, size.height],
      ])
      .on("start", () => svg.classList.add("cursor-grabbing"))
      .on("end", () => svg.classList.remove("cursor-grabbing"))
      .on("zoom", (event) => {
        layerRef.current?.setAttribute("transform", event.transform.toString())
        setScale(event.transform.k)
      })
    zoomRef.current = behaviour
    select(svg).call(behaviour).on("dblclick.zoom", null)
    return () => {
      select(svg).on(".zoom", null)
    }
  }, [size.width, size.height])

  const zoomBy = useCallback((factor: number) => {
    const svg = svgRef.current
    if (svg && zoomRef.current) select(svg).transition().duration(250).call(zoomRef.current.scaleBy, factor)
  }, [])
  const resetZoom = useCallback(() => {
    const svg = svgRef.current
    if (svg && zoomRef.current) select(svg).transition().duration(300).call(zoomRef.current.transform, zoomIdentity)
  }, [])

  const { paths, outline } = useMemo(() => {
    if (!size.width || !size.height) return { paths: [], outline: "" }
    const projection = geoMercator().fitExtent(
      [
        [PAD, PAD],
        [size.width - PAD, size.height - PAD],
      ],
      wards,
    )
    const path = geoPath(projection)
    return {
      paths: wards.features.map((f) => ({
        id: f.properties.ward_id,
        label: f.properties.name ?? f.properties.ward_id,
        d: path(f) ?? "",
      })),
      outline: path(wards) ?? "",
    }
  }, [wards, size])

  const highlighted = paths.find((p) => p.id === highlightedId)

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden">
      <svg
        ref={svgRef}
        role="group"
        aria-label="Ward map. Scroll or pinch to zoom, drag to pan."
        width={size.width}
        height={size.height}
        className="absolute inset-0 block cursor-grab touch-none select-none"
        onPointerLeave={() => onHover(null)}
      >
        <g ref={layerRef}>
          {/* Ward fills, each with its own boundary line. */}
          {paths.map(({ id, label, d }) => {
            const { color, opacity } = fillFor(id)
            return (
              <path
                key={id}
                d={d}
                className="ward"
                style={{ fill: color, fillOpacity: opacity }}
                tabIndex={0}
                aria-label={label}
                onPointerEnter={(e) => onHover({ id, x: e.clientX, y: e.clientY })}
                onPointerMove={(e) => onHover({ id, x: e.clientX, y: e.clientY })}
                onFocus={(e) => {
                  const r = e.currentTarget.getBoundingClientRect()
                  onHover({ id, x: r.left + r.width / 2, y: r.top + r.height / 2 })
                }}
                onBlur={() => onHover(null)}
              />
            )
          })}
          {/* Outer edge of the whole area, slightly heavier than ward lines. */}
          <path d={outline} className="area-outline" />
          {/* Drawn last so the hovered ward's outline sits above its neighbours. */}
          {highlighted && <path d={highlighted.d} className="ward-outline" />}
        </g>
      </svg>

      <div className="absolute top-3 right-3 flex flex-col overflow-hidden rounded-md border bg-card/90 shadow-sm backdrop-blur">
        <Button variant="ghost" size="icon" className="rounded-none" onClick={() => zoomBy(1.6)} disabled={scale >= MAX_ZOOM} aria-label="Zoom in">
          <Plus />
        </Button>
        <Button variant="ghost" size="icon" className="rounded-none border-t" onClick={() => zoomBy(1 / 1.6)} disabled={scale <= 1} aria-label="Zoom out">
          <Minus />
        </Button>
        <Button variant="ghost" size="icon" className="rounded-none border-t" onClick={resetZoom} disabled={scale <= 1} aria-label="Reset zoom">
          <Maximize2 />
        </Button>
      </div>
      {scale > 1 && (
        <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border bg-card/90 px-2 py-0.5 text-xs text-muted-foreground tabular-nums shadow-sm backdrop-blur">
          {scale.toFixed(1)}×
        </div>
      )}
    </div>
  )
}
