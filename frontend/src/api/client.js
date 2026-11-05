// Browser boundary for the operations dashboard. Keep secrets out of this
// file: VITE_AGENT_API_KEY is suitable only for local development; production
// should serve the dashboard behind an authenticated same-origin proxy.
const configuredBaseUrl = import.meta.env.VITE_AGENT_API_URL || "";
const API_BASE_URL = configuredBaseUrl.replace(/\/$/, "");
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || "";

function headers() {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

async function request(path, options = {}) {
  if (!API_BASE_URL) throw new Error("Set VITE_AGENT_API_URL to connect this dashboard.");
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { ...headers(), ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

export const dashboardApiConfigured = Boolean(API_BASE_URL);

export async function fetchHealth() {
  if (!API_BASE_URL) return { status: "demo" };
  return request("/healthz");
}

export function fetchPromptPlan() {
  return request("/v1/dashboard/prompt-plan");
}

export function appendPromptNode({ instruction, priority = 100, condition = {} }) {
  return request("/v1/dashboard/prompt-plan/nodes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instruction, priority, condition }),
  });
}

// SSE is read with fetch rather than EventSource so a development API key can
// be sent in a header. It automatically exits when its AbortSignal is aborted.
export async function streamRuntimeEvents(onEvent, signal) {
  if (!API_BASE_URL) throw new Error("Set VITE_AGENT_API_URL to receive live activity.");
  const response = await fetch(`${API_BASE_URL}/v1/dashboard/events`, {
    headers: { ...headers(), Accept: "text/event-stream" },
    signal,
  });
  if (!response.ok || !response.body) throw new Error(`Live feed unavailable (${response.status})`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (!signal?.aborted) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      for (const frame of frames) {
        const data = frame.split("\n").find((line) => line.startsWith("data: "));
        if (!data) continue;
        try { onEvent(JSON.parse(data.slice(6))); } catch { /* Ignore malformed frames. */ }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export const dashboardDemo = {
  agent: { name: "Relay", state: "Live", uptime: "02h 14m", queue: 3 },
  activeCall: {
    caller: "+91 98••• 4217",
    started: "01:42",
    transcript: "Mujhe kal subah Riya ko call karna yaad dila dena.",
    state: "Listening",
  },
  today: { calls: 18, resolved: 15, review: 3, quality: "94%" },
  reviews: [
    { id: "#142", intent: "other", confidence: "62%", text: "Mujhe papa ko raat mein phone karna hai.", time: "8m ago" },
    { id: "#141", intent: "delivery_address", confidence: "86%", text: "Mera delivery address update karna hai.", time: "22m ago" },
    { id: "#140", intent: "refund_status", confidence: "73%", text: "Mera refund abhi tak aaya nahi.", time: "41m ago" },
  ],
  recent: [
    { name: "Priya S.", topic: "Callback request", outcome: "Resolved", time: "10:14" },
    { name: "Unknown caller", topic: "Delivery address", outcome: "Needs review", time: "09:48" },
  ],
};
