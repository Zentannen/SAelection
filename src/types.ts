import type { Feature, FeatureCollection, MultiPolygon, Polygon } from 'geojson';

export interface WardProperties {
  ward_id: string;
  name?: string;
  municipality?: string;
  /** District (or metro) the ward belongs to. Wards without one only appear under "All districts". */
  district?: string;
  province?: string;
}

export type WardFeature = Feature<Polygon | MultiPolygon, WardProperties>;
export type WardCollection = FeatureCollection<Polygon | MultiPolygon, WardProperties>;

/** Party code -> percentage of the vote (0–100). */
export type Shares = Record<string, number>;

export interface WardPrediction {
  shares: Shares;
  /** 0–1. Used to shade the ward; optional. */
  confidence?: number;
}

export interface PredictionsFile {
  sample?: boolean;
  generated_at?: string;
  wards: Record<string, WardPrediction>;
  /** Optional model output for the whole area. If absent, it is built from the ward predictions. */
  overall?: { shares: Shares };
  /** Optional model output per district (keyed by district name). If absent, built from the wards. */
  districts?: Record<string, { shares: Shares }>;
}

export type ResultStatus = 'not_started' | 'counting' | 'final';

export interface CountUpdate {
  /** ISO timestamp of this results update. */
  at: string;
  /** Cumulative votes per party counted by this time. */
  votes: Record<string, number>;
}

export interface WardResult {
  status: ResultStatus;
  registered?: number;
  /** Party code -> votes counted so far. */
  votes: Record<string, number>;
  /** Optional: cumulative counts at each results update, oldest first (drives the election-night chart). */
  timeline?: CountUpdate[];
}

export interface ResultsFile {
  sample?: boolean;
  updated_at?: string;
  source?: string;
  wards: Record<string, WardResult>;
}

export interface Election {
  /** Key used in `wards` below, e.g. "2021". */
  id: string;
  /** Axis label, e.g. "2021". */
  label: string;
}

/** Past results per ward, already mapped onto the current ward boundaries. */
export interface HistoryFile {
  sample?: boolean;
  /** Oldest first. */
  elections: Election[];
  /** ward_id -> election id -> vote share (0–100) per party. */
  wards: Record<string, Record<string, Shares>>;
}
