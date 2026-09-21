import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  initialState,
  errorState,
  applyReviewDecision,
  escapeHtml,
  type ReviewEvent,
} from "./dashboard.ts";

const ev = (id: string, decision: ReviewEvent["decision"]): ReviewEvent => ({
  id,
  prediction_id: "pred-1",
  decision,
  corrected_attributes: decision === "corrected" ? { brand: "canon" } : null,
  note: null,
  reviewer_id: "rev-1",
  created_at: "2026-01-01T00:00:00Z",
});

describe("dashboard loading/error states", () => {
  it("starts in loading state with no payload", () => {
    const s = initialState();
    assert.equal(s.status, "loading");
    assert.equal(s.payload, null);
    assert.equal(s.error, null);
  });

  it("captures fetch failures in error state", () => {
    const s = errorState("network down");
    assert.equal(s.status, "error");
    assert.equal(s.error, "network down");
    assert.equal(s.payload, null);
  });
});

describe("applyReviewDecision (append-only)", () => {
  it("appends accept/reject/correct events in order", () => {
    let hist: ReviewEvent[] = [];
    hist = applyReviewDecision(hist, ev("1", "accepted"));
    hist = applyReviewDecision(hist, ev("2", "rejected"));
    hist = applyReviewDecision(hist, ev("3", "corrected"));
    assert.deepEqual(
      hist.map((h) => h.decision),
      ["accepted", "rejected", "corrected"],
    );
    assert.deepEqual(hist[2].corrected_attributes, { brand: "canon" });
  });

  it("never mutates the existing history array", () => {
    const before: ReviewEvent[] = [ev("1", "accepted")];
    const frozen = [...before];
    const next = applyReviewDecision(before, ev("2", "rejected"));
    assert.deepEqual(before, frozen); // original untouched
    assert.notEqual(next, before); // new array returned
    assert.equal(next.length, 2);
  });

  it("rejects empty event ids and unknown decisions", () => {
    assert.throws(() => applyReviewDecision([], { ...ev("x", "accepted"), id: "" }), /id/);
    assert.throws(
      () => applyReviewDecision([], ev("x", "maybe" as never)),
      /decision/,
    );
  });
});

describe("escapeHtml (stored-XSS neutralization)", () => {
  it("escapes markup metacharacters", () => {
    assert.equal(
      escapeHtml('<img src=x onerror="alert(1)">'),
      "&lt;img src&#61;x onerror&#61;&quot;alert(1)&quot;&gt;",
    );
  });

  it("renders null/undefined as empty string and leaves plain text intact", () => {
    assert.equal(escapeHtml(null), "");
    assert.equal(escapeHtml(undefined), "");
    assert.equal(escapeHtml("canon R5"), "canon R5");
  });

  it("neutralizes a JSON-stringified correction payload", () => {
    const evil = JSON.stringify({ brand: "</script><script>alert(1)</script>" });
    assert.ok(!/[<>]/.test(escapeHtml(evil)));
  });
});
