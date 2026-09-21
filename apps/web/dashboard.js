// Reviewer dashboard — TypeScript interfaces + pure state helpers (SCAFFOLD).
// Corrections are appended as new review events; the original prediction
// object is never mutated.
/** Loading state shown while the dashboard payload is being fetched. */
export function initialState() {
    return { status: "loading", payload: null, error: null };
}
/** Error state shown when the fetch fails. */
export function errorState(message) {
    return { status: "error", payload: null, error: message };
}
const DECISIONS = ["accepted", "rejected", "corrected"];
/**
 * Append a review event to the history. Returns a NEW array — the original
 * prediction and existing history are never mutated or overwritten.
 */
export function applyReviewDecision(history, event) {
    if (!event.id)
        throw new Error("review event id is required");
    if (!DECISIONS.includes(event.decision)) {
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
export function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"'`/=]/g, (c) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
        "`": "&#96;",
        "/": "&#47;",
        "=": "&#61;",
    })[c]);
}
