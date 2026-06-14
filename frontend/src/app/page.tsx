"use client";

import { useState, useRef, useEffect } from "react";
import api from "../services/api";

// ─── Types ────────────────────────────────────────────────────────────────────

interface TrustComponents {
  derivability?: number;
  retrieval_confidence?: number;
  context_coherence?: number;
  rule_adjustment?: number;
}

interface TrustInfo {
  score: number;
  label: string;
  color: string;
  components?: TrustComponents;
}

interface DerivabilityBreakdown {
  keyword_coverage?: number;
  question_coverage?: number;
  chunk_utilization?: number;
  contributing_chunks?: number;
  total_chunks?: number;
}

interface DerivabilityInfo {
  score: number;
  is_derivable: boolean;
  breakdown?: DerivabilityBreakdown;
}

interface RetrievalInfo {
  scores: number[];
  avg_score: number;
  chunks_used: number;
}

interface RulesInfo {
  passed: boolean;
  flagged_keywords: string[];
  out_of_scope: boolean;
  medical_relevance: boolean;
  warnings: string[];
  notes: string;
}

interface SessionStats {
  total_queries: number;
  avg_trust_score?: number;
  max_trust_score?: number;
  min_trust_score?: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  trust?: TrustInfo;
  derivability?: DerivabilityInfo;
  retrieval?: RetrievalInfo;
  rules?: RulesInfo;
  abstained?: boolean;
  timestamp: Date;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function TrustBadge({ label, color }: { label: string; color: string }) {
  const styles: Record<string, React.CSSProperties> = {
    green:  { background: "rgba(16,185,129,0.15)", color: "#6ee7b7", border: "1px solid rgba(16,185,129,0.3)" },
    amber:  { background: "rgba(245,158,11,0.15)", color: "#fcd34d", border: "1px solid rgba(245,158,11,0.3)" },
    orange: { background: "rgba(249,115,22,0.15)", color: "#fdba74", border: "1px solid rgba(249,115,22,0.3)" },
    red:    { background: "rgba(239,68,68,0.15)",  color: "#fca5a5", border: "1px solid rgba(239,68,68,0.3)"  },
    gray:   { background: "rgba(100,116,139,0.15)",color: "#94a3b8", border: "1px solid rgba(100,116,139,0.3)"},
  };
  return (
    <span style={{ ...styles[color] ?? styles.gray, fontSize: 11, fontWeight: 600, padding: "2px 8px", borderRadius: 20 }}>
      {label}
    </span>
  );
}

function ScoreBar({ value, max = 100, color }: { value: number; max?: number; color: string }) {
  const pct = Math.min(100, Math.round((value / max) * 100));
  const colorMap: Record<string, string> = {
    green: "#10b981", amber: "#f59e0b", orange: "#f97316",
    red: "#ef4444", blue: "#3b82f6", purple: "#8b5cf6", gray: "#64748b",
  };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%", borderRadius: 3,
          background: colorMap[color] ?? colorMap.gray,
          transition: "width 0.7s ease",
        }} />
      </div>
      <span style={{ fontSize: 11, color: "#94a3b8", minWidth: 28, textAlign: "right" }}>{Math.round(value)}</span>
    </div>
  );
}

function ScorePanel({
  trust, derivability, retrieval, rules,
}: {
  trust?: TrustInfo;
  derivability?: DerivabilityInfo;
  retrieval?: RetrievalInfo;
  rules?: RulesInfo;
}) {
  const [open, setOpen] = useState(false);
  if (!trust) return null;

  return (
    <div style={{ marginTop: 6, borderRadius: 10, border: "1px solid rgba(255,255,255,0.08)", overflow: "hidden" }}>
      {/* Collapsed header */}
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "8px 12px", background: "rgba(15,23,42,0.6)", cursor: "pointer", border: "none",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1 }}>
          <span style={{ fontSize: 11, color: "#64748b", fontWeight: 600, whiteSpace: "nowrap" }}>Trust Score</span>
          <div style={{ flex: 1, maxWidth: 120 }}>
            <ScoreBar value={trust.score} color={trust.color} />
          </div>
          <TrustBadge label={trust.label} color={trust.color} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {rules && (
            <span style={{
              width: 8, height: 8, borderRadius: "50%",
              background: rules.passed ? "#10b981" : "#ef4444",
              flexShrink: 0,
            }} title={rules.passed ? "Rules passed" : "Rule warnings"} />
          )}
          <svg width={14} height={14} viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth={2}
            style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expanded detail */}
      {open && (
        <div style={{
          padding: "12px 14px", background: "rgba(2,6,23,0.5)",
          display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px 20px",
        }}>
          {/* Derivability */}
          {derivability && (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600 }}>📐 Derivability</span>
                <span style={{ fontSize: 11, fontWeight: 700, color: derivability.is_derivable ? "#6ee7b7" : "#fca5a5" }}>
                  {(derivability.score * 100).toFixed(1)}%
                </span>
              </div>
              {derivability.breakdown && (
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {[
                    ["Keyword coverage", ((derivability.breakdown.keyword_coverage ?? 0) * 100).toFixed(0) + "%"],
                    ["Question coverage", ((derivability.breakdown.question_coverage ?? 0) * 100).toFixed(0) + "%"],
                    ["Chunks used", `${derivability.breakdown.contributing_chunks}/${derivability.breakdown.total_chunks}`],
                  ].map(([label, val]) => (
                    <div key={label} style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ fontSize: 11, color: "#475569" }}>{label}</span>
                      <span style={{ fontSize: 11, color: "#94a3b8" }}>{val}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Trust Components */}
          {trust.components && (
            <div>
              <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, marginBottom: 6 }}>🛡️ Components</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {[
                  ["Derivability", trust.components.derivability ?? 0, "blue"],
                  ["Retrieval", trust.components.retrieval_confidence ?? 0, "purple"],
                  ["Coherence", trust.components.context_coherence ?? 0, "amber"],
                ] .map(([label, val, color]) => (
                  <div key={label as string}>
                    <div style={{ fontSize: 10, color: "#475569", marginBottom: 2 }}>{label as string}</div>
                    <ScoreBar value={val as number} color={color as string} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Retrieval */}
          {retrieval && (
            <div>
              <div style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600, marginBottom: 6 }}>🔍 Retrieval</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {[
                  ["Chunks used", retrieval.chunks_used],
                  ["Avg similarity", (retrieval.avg_score * 100).toFixed(1) + "%"],
                ].map(([label, val]) => (
                  <div key={label as string} style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 11, color: "#475569" }}>{label as string}</span>
                    <span style={{ fontSize: 11, color: "#94a3b8" }}>{val as string | number}</span>
                  </div>
                ))}
                {retrieval.scores.length > 0 && (
                  <div style={{ marginTop: 4 }}>
                    <div style={{ fontSize: 10, color: "#475569", marginBottom: 3 }}>Per-chunk scores</div>
                    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                      {retrieval.scores.map((s, i) => (
                        <span key={i} style={{
                          fontSize: 10, padding: "2px 6px", borderRadius: 4,
                          background: "rgba(255,255,255,0.06)", color: "#94a3b8",
                        }}>
                          {(s * 100).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Rule Filter */}
          {rules && (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 600 }}>⚖️ Rule Filter</span>
                <span style={{ fontSize: 11, fontWeight: 700, color: rules.passed ? "#6ee7b7" : "#fdba74" }}>
                  {rules.passed ? "Passed" : "Warned"}
                </span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {[
                  ["Medical relevance", rules.medical_relevance ? "✓" : "✗", rules.medical_relevance ? "#6ee7b7" : "#fca5a5"],
                  ["Out of scope", rules.out_of_scope ? "Yes" : "No", rules.out_of_scope ? "#fca5a5" : "#6ee7b7"],
                ].map(([label, val, color]) => (
                  <div key={label as string} style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 11, color: "#475569" }}>{label as string}</span>
                    <span style={{ fontSize: 11, color: color as string }}>{val as string}</span>
                  </div>
                ))}
                {rules.flagged_keywords.length > 0 && (
                  <div style={{ fontSize: 11, color: "#fdba74" }}>
                    Flagged: {rules.flagged_keywords.join(", ")}
                  </div>
                )}
                {rules.warnings.map((w, i) => (
                  <div key={i} style={{ fontSize: 10, color: "#f97316" }}>⚠ {w}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

const SUGGESTIONS = [
  "What are the symptoms of type 2 diabetes?",
  "How is hypertension treated?",
  "What causes chronic kidney disease?",
  "Explain the stages of heart failure",
];

export default function Home() {
  const [question, setQuestion] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [sessionStats, setSessionStats] = useState<SessionStats | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendQuestion = async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed || loading) return;

    setMessages(prev => [...prev, { role: "user", content: trimmed, timestamp: new Date() }]);
    setQuestion("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    setLoading(true);

    try {
      const { data } = await api.post("/chat/", { question: trimmed });

      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.answer ?? "No answer returned.",
        trust: data.trust,
        derivability: data.derivability,
        retrieval: data.retrieval,
        rules: data.rules,
        abstained: data.abstained,
        timestamp: new Date(),
      }]);

      if (data.session_stats) setSessionStats(data.session_stats);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `Connection error: ${msg}. Make sure the Django server is running on port 8000.`,
        timestamp: new Date(),
      }]);
    } finally {
      setLoading(false);
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion(question);
    }
  };

  const clearChat = async () => {
    setMessages([]);
    setSessionStats(null);
    try { await api.delete("/chat/clear/"); } catch { /* ignore */ }
  };

  // ── Styles (inline to avoid Tailwind compilation issues) ─────────────────
  const page: React.CSSProperties = {
    minHeight: "100vh", display: "flex", flexDirection: "column",
    background: "linear-gradient(160deg, #060d1f 0%, #0a1628 50%, #060d1f 100%)",
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    color: "#e2e8f0",
  };

  const header: React.CSSProperties = {
    borderBottom: "1px solid rgba(255,255,255,0.06)",
    background: "rgba(10,22,40,0.7)",
    backdropFilter: "blur(12px)",
    position: "sticky", top: 0, zIndex: 10,
  };

  const headerInner: React.CSSProperties = {
    maxWidth: 800, margin: "0 auto", padding: "12px 20px",
    display: "flex", alignItems: "center", justifyContent: "space-between",
  };

  const main: React.CSSProperties = {
    flex: 1, maxWidth: 800, width: "100%", margin: "0 auto",
    padding: "24px 20px", display: "flex", flexDirection: "column", gap: 20,
  };

  const footer: React.CSSProperties = {
    borderTop: "1px solid rgba(255,255,255,0.06)",
    background: "rgba(10,22,40,0.7)",
    backdropFilter: "blur(12px)",
    position: "sticky", bottom: 0,
  };

  return (
    <div style={page}>
      {/* ── Header ── */}
      <header style={header}>
        <div style={headerInner}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: "rgba(59,130,246,0.15)", border: "1px solid rgba(59,130,246,0.3)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18,
            }}>⚕️</div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15, color: "#f1f5f9", letterSpacing: "-0.02em" }}>
                Healthcare KIS
              </div>
              <div style={{ fontSize: 11, color: "#475569" }}>
                Knowledge Infrastructure System
              </div>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            {sessionStats && sessionStats.total_queries > 0 && (
              <div style={{ fontSize: 11, color: "#475569", display: "flex", gap: 12 }}>
                <span>{sessionStats.total_queries} queries</span>
                {sessionStats.avg_trust_score !== undefined && (
                  <span>Avg trust: <strong style={{ color: "#94a3b8" }}>{sessionStats.avg_trust_score}</strong></span>
                )}
              </div>
            )}
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                style={{
                  fontSize: 12, color: "#475569", cursor: "pointer",
                  padding: "4px 10px", borderRadius: 6, border: "none",
                  background: "transparent",
                }}
                onMouseEnter={e => (e.currentTarget.style.color = "#94a3b8")}
                onMouseLeave={e => (e.currentTarget.style.color = "#475569")}
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ── Chat ── */}
      <main style={main}>
        {/* Welcome screen */}
        {messages.length === 0 && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "60px 0", gap: 24, textAlign: "center" }}>
            <div style={{
              width: 72, height: 72, borderRadius: 20,
              background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.2)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 36,
            }}>⚕️</div>
            <div>
              <h2 style={{ fontSize: 22, fontWeight: 700, color: "#f1f5f9", margin: "0 0 8px", letterSpacing: "-0.02em" }}>
                Medical Knowledge Assistant
              </h2>
              <p style={{ fontSize: 14, color: "#475569", maxWidth: 380, margin: 0, lineHeight: 1.6 }}>
                Ask questions about conditions, treatments, and protocols. Every answer is scored for trust and derivability.
              </p>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, width: "100%", maxWidth: 440 }}>
              {SUGGESTIONS.map(s => (
                <button
                  key={s}
                  onClick={() => sendQuestion(s)}
                  style={{
                    textAlign: "left", fontSize: 12, color: "#64748b",
                    border: "1px solid rgba(255,255,255,0.07)", borderRadius: 10,
                    padding: "10px 12px", background: "transparent", cursor: "pointer",
                    lineHeight: 1.4, transition: "all 0.15s",
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.color = "#cbd5e1";
                    e.currentTarget.style.background = "rgba(255,255,255,0.04)";
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.12)";
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.color = "#64748b";
                    e.currentTarget.style.background = "transparent";
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.07)";
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, i) => (
          <div key={i} style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
            <div style={{ maxWidth: msg.role === "user" ? "68%" : "100%", width: msg.role === "assistant" ? "100%" : undefined }}>
              <div style={{
                borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "4px 18px 18px 18px",
                padding: "12px 16px",
                background: msg.role === "user"
                  ? "linear-gradient(135deg, #2563eb, #1d4ed8)"
                  : "rgba(255,255,255,0.04)",
                border: msg.role === "assistant" ? "1px solid rgba(255,255,255,0.07)" : "none",
                fontSize: 14, lineHeight: 1.65,
                color: msg.role === "user" ? "#fff" : "#cbd5e1",
              }}>
                {msg.abstained && (
                  <div style={{ fontSize: 12, color: "#f97316", marginBottom: 6, fontWeight: 600 }}>
                    ⚠️ Low confidence — abstained from answering
                  </div>
                )}
                <p style={{ margin: 0, whiteSpace: "pre-wrap" }}>{msg.content}</p>
                <div style={{ fontSize: 11, opacity: 0.4, marginTop: 6 }}>
                  {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </div>
              </div>

              {/* Score panel — assistant only */}
              {msg.role === "assistant" && (msg.trust ?? msg.rules) && (
                <ScorePanel
                  trust={msg.trust}
                  derivability={msg.derivability}
                  retrieval={msg.retrieval}
                  rules={msg.rules}
                />
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div style={{ display: "flex" }}>
            <div style={{
              background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)",
              borderRadius: "4px 18px 18px 18px", padding: "14px 18px",
              display: "flex", alignItems: "center", gap: 10,
            }}>
              <div style={{ display: "flex", gap: 4 }}>
                {[0, 1, 2].map(i => (
                  <span key={i} style={{
                    width: 6, height: 6, borderRadius: "50%", background: "#3b82f6",
                    display: "inline-block",
                    animation: "bounce 1.2s infinite",
                    animationDelay: `${i * 0.2}s`,
                  }} />
                ))}
              </div>
              <span style={{ fontSize: 12, color: "#475569" }}>Searching knowledge base…</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* ── Input ── */}
      <footer style={footer}>
        <div style={{ maxWidth: 800, margin: "0 auto", padding: "12px 20px" }}>
          <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
            <textarea
              ref={textareaRef}
              value={question}
              onChange={e => {
                setQuestion(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 130)}px`;
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask a medical question… (Enter to send, Shift+Enter for newline)"
              disabled={loading}
              rows={1}
              style={{
                flex: 1, borderRadius: 14, padding: "11px 16px",
                background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
                color: "#e2e8f0", fontSize: 14, resize: "none", lineHeight: 1.5,
                outline: "none", fontFamily: "inherit",
                transition: "border-color 0.15s",
              }}
              onFocus={e => (e.currentTarget.style.borderColor = "rgba(59,130,246,0.5)")}
              onBlur={e => (e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)")}
            />
            <button
              onClick={() => sendQuestion(question)}
              disabled={!question.trim() || loading}
              style={{
                width: 42, height: 42, borderRadius: 12, border: "none",
                background: question.trim() && !loading ? "#2563eb" : "rgba(255,255,255,0.08)",
                color: question.trim() && !loading ? "#fff" : "#475569",
                cursor: question.trim() && !loading ? "pointer" : "not-allowed",
                display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0, transition: "all 0.15s",
              }}
            >
              <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
          <p style={{ fontSize: 11, color: "#334155", textAlign: "center", margin: "8px 0 0" }}>
            AI-generated from your knowledge base · Always consult a qualified healthcare professional
          </p>
        </div>
      </footer>

      <style>{`
        * { box-sizing: border-box; }
        body { margin: 0; }
        ::placeholder { color: #334155; }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-6px); opacity: 1; }
        }
      `}</style>
    </div>
  );
}