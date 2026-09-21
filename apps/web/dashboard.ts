// Reviewer dashboard — TypeScript interfaces + pure state helpers (SCAFFOLD).
// Corrections are appended as new review events; the original prediction
// object is never mutated.

export interface DashboardImage {
  id: string;
  storage_key: string;
  filename: string;
  content_type: string;
  byte_size: number;
  width: number | null;
  height: number | null;
  sha256: string;
}

export interface DashboardEvidence {
  id: string;
  source: "text" | "image" | "multimodal";
  image_id: string | null;
  text_span: { field: string; start: number; end: number } | null;
  bounding_box: { x: number; y: number; w: number; h: number } | null;
  note: string | null;
}

export interface DashboardAttribute {
  id: string;
  key: string;
  value: string | null;
  confidence: number;
  abstained: boolean;
  evidence: DashboardEvidence[];
}

export interface DashboardPrediction {
  id: string;
  prediction_type: string;
  latency_ms: number | null;
  created_at: string;
  model_version: { id: string; name: string; version: string };
  attributes: DashboardAttribute[];
}

export interface DuplicateMatch {
  prediction_id: string;
  candidate_listing_id: string;
  score: number;
}

export interface PriceInterval {
  low: number;
  high: number;
}

export interface RiskFinding {
  risk_category: string;
  severity: "low" | "medium" | "high";
  confidence: number;
  message: string;
  recommended_action: "publish" | "warn" | "human_review";
}

export interface ReviewEvent {
  id: string;
  prediction_id: string | null;
  decision: "accepted" | "rejected" | "corrected";
  corrected_attributes: Record<string, string | null> | null;
  note: string | null;
  reviewer_id: string;
  created_at: string;
}

export interface DashboardPayload {
  listing: {
    id: string;
    seller_id: string;
    title: string;
    description: string | null;
    category: string | null;
    status: string;
  };
  images: DashboardImage[];
  predictions: DashboardPrediction[];
  duplicate_matches: DuplicateMatch[];
  price_interval: PriceInterval | null;
  risk_findings: RiskFinding[];
  decision_history: ReviewEvent[];
}

export type DashboardStatus = "loading" | "ready" | "error";

export interface DashboardState {
  status: DashboardStatus;
  payload: DashboardPayload | null;
  error: string | null;
}

/** Loading state shown while the dashboard payload is being fetched. */
export function initialState(): DashboardState {
  return { status: "loading", payload: null, error: null };
}

/** Error state shown when the fetch fails. */
export function errorState(message: string): DashboardState {
  return { status: "error", payload: null, error: message };
}

const DECISIONS = ["accepted", "rejected", "corrected"] as const;

/**
 * Append a review event to the history. Returns a NEW array — the original
 * prediction and existing history are never mutated or overwritten.
 */
export function applyReviewDecision(
  history: ReviewEvent[],
  event: ReviewEvent,
): ReviewEvent[] {
  if (!event.id) throw new Error("review event id is required");
  if (!(DECISIONS as readonly string[]).includes(event.decision)) {
    throw new Error(`unknown decision: ${event.decision}`);
  }
  return [...history, event];
}

/**
 * Escape a value for interpolation into innerHTML. The API returns
 * seller-controlled strings verbatim by design (filename, attribute
 * key/value, reviewer_id, corrected_attributes); this is the single
 * neutralization point before they reach the DOM. Numbers are safe
 * as-is; null/undefined render as empty string.
 */
export function escapeHtml(value: unknown): string {
  return String(value ?? "").replace(
    /[&<>"'`/=]/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
        "`": "&#96;",
        "/": "&#47;",
        "=": "&#61;",
      })[c]!,
  );
}
