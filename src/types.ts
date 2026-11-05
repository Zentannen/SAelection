import type { Feature, FeatureCollection, MultiPolygon, Polygon } from 'geojson';

export interface WardProperties {
  ward_id: string;
  name?: string;
  municipality?: string;
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
}

export type ResultStatus = 'not_started' | 'counting' | 'final';

export interface WardResult {
  status: ResultStatus;
  registered?: number;
  /** Party code -> votes counted so far. */
  votes: Record<string, number>;
}

export interface ResultsFile {
  sample?: boolean;
  updated_at?: string;
  source?: string;
  wards: Record<string, WardResult>;
}
