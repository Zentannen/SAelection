import { topology } from "topojson-server"
import { mesh } from "topojson-client"
import type { GeometryCollection, Topology } from "topojson-specification"
import type { MultiLineString } from "geojson"
import type { WardCollection, WardProperties } from "@/types"

type WardTopology = Topology<{ wards: GeometryCollection<WardProperties> }>

/**
 * District borders are derived from the ward boundaries, so no separate
 * district file is needed. Converting to TopoJSON finds the edges neighbouring
 * wards share; a district border is any shared edge whose two wards are in
 * different districts.
 *
 * Quantising to 1e6 steps snaps near-identical vertices together (≈1 m across
 * a province), so borders still line up when two neighbouring ward shapes
 * disagree in the last decimal place.
 */
export function buildDistrictGeometry(wards: WardCollection) {
  const topo = topology({ wards }, 1e6) as unknown as WardTopology
  const object = topo.objects.wards
  const districtOf = (g: { properties?: object | null }) => (g.properties as Partial<WardProperties> | undefined)?.district || null

  /** Borders between two different districts (not the outer edge of the map). */
  const borders: MultiLineString = mesh(topo, object, (a, b) => a !== b && districtOf(a) !== districtOf(b))

  const outlineCache = new Map<string, MultiLineString>()
  /** Full outline of one district, including any stretch on the outer edge. */
  const outline = (district: string): MultiLineString => {
    let line = outlineCache.get(district)
    if (!line) {
      line = mesh(topo, object, (a, b) =>
        a === b ? districtOf(a) === district : (districtOf(a) === district) !== (districtOf(b) === district),
      )
      outlineCache.set(district, line)
    }
    return line
  }

  return { borders, outline }
}
