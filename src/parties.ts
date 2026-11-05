// Display config per party code. The top ten parties get their own colour;
// everyone else shares "Other". Ten hues can't all be told apart by colour
// alone (especially with colour-blindness), so names are always shown next to
// colours and the map legend can highlight one party's wards.
// `ink` is the text colour for a label printed inside that party's fill.
export const PARTIES: Record<string, { name: string; color?: string; ink?: string }> = {
  // The ten largest parties (2024 national vote order) get their own colour.
  ANC: { name: "African National Congress", color: "var(--party-anc)", ink: "#ffffff" },
  DA: { name: "Democratic Alliance", color: "var(--party-da)", ink: "#ffffff" },
  MK: { name: "uMkhonto weSizwe", color: "var(--party-mk)", ink: "#ffffff" },
  EFF: { name: "Economic Freedom Fighters", color: "var(--party-eff)", ink: "#ffffff" },
  IFP: { name: "Inkatha Freedom Party", color: "var(--party-ifp)", ink: "#0b0b0b" },
  PA: { name: "Patriotic Alliance", color: "var(--party-pa)", ink: "#ffffff" },
  "FF+": { name: "Freedom Front Plus", color: "var(--party-ffplus)", ink: "#ffffff" },
  ActionSA: { name: "ActionSA", color: "var(--party-actionsa)", ink: "#0b0b0b" },
  ACDP: { name: "African Christian Democratic Party", color: "var(--party-acdp)", ink: "#ffffff" },
  UDM: { name: "United Democratic Movement", color: "var(--party-udm)", ink: "#0b0b0b" },
}

export const OTHER = "Other"
export const OTHER_COLOR = "var(--party-other)"
const OTHER_INK = "var(--party-other-ink)"

/** Parties with their own colour, in legend order. */
export const MAIN_PARTIES = Object.keys(PARTIES).filter((p) => PARTIES[p].color)

export const partyColor = (code: string) => PARTIES[code]?.color ?? OTHER_COLOR
export const partyInk = (code: string) => (PARTIES[code]?.color ? PARTIES[code].ink : undefined) ?? OTHER_INK
export const partyName = (code: string) => (code === OTHER ? "Other parties" : (PARTIES[code]?.name ?? code))

/** Map a party code to itself if it has its own colour, otherwise to "Other". */
export const colourBucket = (code: string) => (PARTIES[code]?.color ? code : OTHER)
