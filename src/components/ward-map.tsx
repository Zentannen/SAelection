import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { geoMercator, geoPath } from "d3-geo"
import { select } from "d3-selection"
import "d3-transition"
import { zoom, zoomIdentity, type ZoomBehavior } from "d3-zoom"
import { ArrowLeft, Maximize2, Minus, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { buildDistrictGeometry } from "@/lib/geo"
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
  /** Ward whose detail panel is open; stays outlined. */
  selectedId?: string | null
  onHover: (e: HoverEvent | null) => void
  /** Click / Enter on a ward. */
  onSelectWard?: (id: string) => void
  /** District the map is zoomed to; other districts are dimmed. */
  focusDistrict?: string | null
  /** District to outline (e.g. while its row is hovered in the list). */
  highlightDistrict?: string | null
  onSelectDistrict?: (district: string | null) => void
}

const PAD = 12
const MAX_ZOOM = 16

export function WardMap({
  wards,
  fillFor,
  highlightedId,
  selectedId,
  onHover,
  onSelectWard,
  focusDistrict = null,
  highlightDistrict = null,
  onSelectDistrict,
}: Props) {
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
  // <g> (no React re-render per frame), so it stays smooth with thousands of
  // wards. --k lets labels and outlines counter-scale to a constant size.
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
        svg.style.setProperty("--k", String(event.transform.k))
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

  // Topology work only depends on the shapes, not the screen size.
  const districtGeometry = useMemo(() => buildDistrictGeometry(wards), [wards])

  const geo = useMemo(() => {
    if (!size.width || !size.height) return null
    const projection = geoMercator().fitExtent(
      [
        [PAD, PAD],
        [size.width - PAD, size.height - PAD],
      ],
      wards,
    )
    const path = geoPath(projection)
    const paths = wards.features.map((f) => ({
      id: f.properties.ward_id,
      label: f.properties.name ?? f.properties.ward_id,
      district: f.properties.district || null,
      d: path(f) ?? "",
      bounds: path.bounds(f),
      centroid: path.centroid(f),
      area: path.area(f),
    }))

    // District label sits at the area-weighted centre of its wards.
    const districts = new Map<string, { x: number; y: number; area: number; x0: number; y0: number; x1: number; y1: number }>()
    for (const p of paths) {
      if (!p.district || !Number.isFinite(p.centroid[0])) continue
      const d = districts.get(p.district) ?? { x: 0, y: 0, area: 0, x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity }
      d.x += p.centroid[0] * p.area
      d.y += p.centroid[1] * p.area
      d.area += p.area
      d.x0 = Math.min(d.x0, p.bounds[0][0])
      d.y0 = Math.min(d.y0, p.bounds[0][1])
      d.x1 = Math.max(d.x1, p.bounds[1][0])
      d.y1 = Math.max(d.y1, p.bounds[1][1])
      districts.set(p.district, d)
    }
    const districtInfo = [...districts.entries()].map(([name, d]) => ({
      name,
      x: d.area ? d.x / d.area : (d.x0 + d.x1) / 2,
      y: d.area ? d.y / d.area : (d.y0 + d.y1) / 2,
      bounds: [
        [d.x0, d.y0],
        [d.x1, d.y1],
      ] as [[number, number], [number, number]],
    }))

    return {
      path,
      paths,
      districtInfo,
      outline: path(wards) ?? "",
      borders: path(districtGeometry.borders) ?? "",
    }
  }, [wards, size, districtGeometry])

  // Zoom to the focused district (or back out to the whole map).
  useEffect(() => {
    const svg = svgRef.current
    if (!svg || !zoomRef.current || !geo) return
    const target = focusDistrict ? geo.districtInfo.find((d) => d.name === focusDistrict) : null
    let transform = zoomIdentity
    if (target) {
      const [[x0, y0], [x1, y1]] = target.bounds
      const k = Math.min(MAX_ZOOM, 0.88 / Math.max((x1 - x0) / size.width, (y1 - y0) / size.height))
      transform = zoomIdentity
        .translate(size.width / 2, size.height / 2)
        .scale(Math.max(1, k))
        .translate(-(x0 + x1) / 2, -(y0 + y1) / 2)
    }
    select(svg).transition().duration(650).call(zoomRef.current.transform, transform)
  }, [focusDistrict, geo, size.width, size.height])

  // Escape leaves district focus.
  useEffect(() => {
    if (!focusDistrict || !onSelectDistrict) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onSelectDistrict(null)
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [focusDistrict, onSelectDistrict])

  const highlighted = highlightedId !== selectedId ? geo?.paths.find((p) => p.id === highlightedId) : undefined
  const selected = geo?.paths.find((p) => p.id === selectedId)
  const focusOutline = useMemo(
    () => (geo && focusDistrict ? (geo.path(districtGeometry.outline(focusDistrict)) ?? "") : ""),
    [geo, focusDistrict, districtGeometry],
  )
  const highlightOutline = useMemo(
    () =>
      geo && highlightDistrict && highlightDistrict !== focusDistrict
        ? (geo.path(districtGeometry.outline(highlightDistrict)) ?? "")
        : "",
    [geo, highlightDistrict, focusDistrict, districtGeometry],
  )

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden">
      <svg
        ref={svgRef}
        role="group"
        aria-label="Ward map. Scroll or pinch to zoom, drag to pan, click a ward for its details."
        width={size.width}
        height={size.height}
        className="absolute inset-0 block cursor-grab touch-none select-none"
        onPointerLeave={() => onHover(null)}
      >
        <g ref={layerRef}>
          {/* Ward fills, each with its own boundary line. */}
          {geo?.paths.map(({ id, label, d, district }) => {
            const { color, opacity } = fillFor(id)
            const dimmed = !!focusDistrict && district !== focusDistrict
            return (
              <path
                key={id}
                d={d}
                className={dimmed ? "ward dimmed" : "ward"}
                style={{ fill: color, fillOpacity: opacity }}
                tabIndex={0}
                aria-label={district ? `${label}, ${district}` : label}
                onPointerEnter={(e) => onHover({ id, x: e.clientX, y: e.clientY })}
                onPointerMove={(e) => onHover({ id, x: e.clientX, y: e.clientY })}
                onClick={() => onSelectWard?.(id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    onSelectWard?.(id)
                  }
                }}
                onFocus={(e) => {
                  const r = e.currentTarget.getBoundingClientRect()
                  onHover({ id, x: r.left + r.width / 2, y: r.top + r.height / 2 })
                }}
                onBlur={() => onHover(null)}
              />
            )
          })}
          {/* District borders: heavier than ward lines. */}
          {geo && <path d={geo.borders} className="district-border" />}
          {/* Outer edge of the whole area. */}
          {geo && <path d={geo.outline} className="area-outline" />}
          {focusOutline && <path d={focusOutline} className="district-focus" />}
          {highlightOutline && <path d={highlightOutline} className="district-highlight" />}
          {/* Drawn last so ward outlines sit above their neighbours. */}
          {selected && <path d={selected.d} className="ward-selected" />}
          {highlighted && <path d={highlighted.d} className="ward-outline" />}
          {geo?.districtInfo.map((d) => (
            <text
              key={d.name}
              x={d.x}
              y={d.y}
              className={focusDistrict && focusDistrict !== d.name ? "district-label dimmed" : "district-label"}
              textAnchor="middle"
              dominantBaseline="central"
            >
              {d.name}
            </text>
          ))}
        </g>
      </svg>

      {focusDistrict && onSelectDistrict && (
        <div className="absolute top-3 left-3">
          <Button variant="outline" size="sm" className="bg-card/90 shadow-sm backdrop-blur" onClick={() => onSelectDistrict(null)}>
            <ArrowLeft /> All districts
          </Button>
        </div>
      )}

      <div className="absolute top-3 right-3 flex flex-col overflow-hidden rounded-md border bg-card/90 shadow-sm backdrop-blur">
        <Button variant="ghost" size="icon" className="rounded-none" onClick={() => zoomBy(1.6)} disabled={scale >= MAX_ZOOM} aria-label="Zoom in">
          <Plus />
        </Button>
        <Button variant="ghost" size="icon" className="rounded-none border-t" onClick={() => zoomBy(1 / 1.6)} disabled={scale <= 1} aria-label="Zoom out">
          <Minus />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="rounded-none border-t"
          onClick={() => (focusDistrict && onSelectDistrict ? onSelectDistrict(null) : resetZoom())}
          disabled={scale <= 1 && !focusDistrict}
          aria-label="Reset zoom"
        >
          <Maximize2 />
        </Button>
      </div>
      {scale > 1.05 && (
        <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border bg-card/90 px-2 py-0.5 text-xs text-muted-foreground tabular-nums shadow-sm backdrop-blur">
          {scale.toFixed(1)}×
        </div>
      )}
    </div>
  )
}
