import { useEffect, useMemo, useRef, useState } from "react";
import { appendPromptNode, dashboardApiConfigured, dashboardDemo, fetchPromptPlan, streamRuntimeEvents } from "./api/client";

const iconPaths = {
  home: <><path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1Z" /></>,
  calls: <><path d="M20 16.3v2.2a1.5 1.5 0 0 1-1.6 1.5A16.8 16.8 0 0 1 4 5.6 1.5 1.5 0 0 1 5.5 4H8a1 1 0 0 1 1 .8l.8 3a1 1 0 0 1-.3 1L8 10.3a13 13 0 0 0 5.7 5.7l1.5-1.5a1 1 0 0 1 1-.3l3 .8a1 1 0 0 1 .8 1Z" /></>,
  review: <><path d="M5 4h14v16H5z" /><path d="M8 8h8M8 12h8M8 16h4" /><path d="m16.5 17 1.2 1.2 2.8-3" /></>,
  settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.1 2.1-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5v.2h-3v-.2a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.9.3l-.1.1-2.1-2.1.1-.1A1.7 1.7 0 0 0 7 15a1.7 1.7 0 0 0-1.5-1H5.3v-3h.2A1.7 1.7 0 0 0 7 10a1.7 1.7 0 0 0-.3-1.9l-.1-.1 2.1-2.1.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.5v-.2h3v.2a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1 2.1 2.1-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.5 1h.2v3h-.2a1.7 1.7 0 0 0-1.4 1Z" /></>,
  bell: <><path d="M18 9a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4" /></>,
  chevron: <path d="m9 18 6-6-6-6" />,
  play: <><path d="m8 5 10 7-10 7Z" /></>,
  pause: <><path d="M8 5v14M16 5v14" /></>,
  check: <path d="m5 12 4.2 4L19 6.5" />,
  sliders: <><path d="M4 6h16M4 18h16M9 6v0M15 18v0" /><circle cx="9" cy="6" r="2" /><circle cx="15" cy="18" r="2" /></>,
  branch: <><path d="M6 3v12a4 4 0 0 0 4 4h8" /><path d="M14 5h4v4M14 19h4v-4" /><path d="M10 9h8" /></>,
  plug: <><path d="M9 7V3M15 7V3M7 7h10v3a5 5 0 0 1-10 0Z" /><path d="M12 15v6" /></>,
  database: <><ellipse cx="12" cy="5" rx="7" ry="3" /><path d="M5 5v7c0 1.7 3.1 3 7 3s7-1.3 7-3V5M5 12v7c0 1.7 3.1 3 7 3s7-1.3 7-3v-7" /></>,
  shield: <path d="M12 3 19 6v5c0 4.4-2.8 8-7 10-4.2-2-7-5.6-7-10V6Z" />,
  plus: <><path d="M12 5v14M5 12h14" /></>,
  eye: <><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" /><circle cx="12" cy="12" r="2.5" /></>,
  refresh: <><path d="M20 11a8 8 0 1 0 2 5.5" /><path d="M20 4v7h-7" /></>,
  link: <><path d="M10 13a5 5 0 0 0 7.1.1l2.3-2.3a5 5 0 0 0-7.1-7.1l-1.3 1.3" /><path d="M14 11a5 5 0 0 0-7.1-.1l-2.3 2.3a5 5 0 0 0 7.1 7.1l1.3-1.3" /></>,
  more: <><circle cx="5" cy="12" r=".8" fill="currentColor" /><circle cx="12" cy="12" r=".8" fill="currentColor" /><circle cx="19" cy="12" r=".8" fill="currentColor" /></>,
};

function Icon({ name, size = 20 }) {
  return <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{iconPaths[name]}</svg>;
}

function StatusPill({ children, tone = "mint" }) {
  return <span className={`status-pill ${tone}`}><i />{children}</span>;
}

function App() {
  const [activeTab, setActiveTab] = useState(() => new URLSearchParams(window.location.search).get("tab") || "home");
  const [callLive, setCallLive] = useState(true);
  const [reviewed, setReviewed] = useState(new Set());
  const [reviewedLabels, setReviewedLabels] = useState({});
  const [selectedReview, setSelectedReview] = useState(null);
  const [lastReviewed, setLastReviewed] = useState(null);
  const [notice, setNotice] = useState("");
  const [instructionNodes, setInstructionNodes] = useState(defaultInstructionNodes);
  const [preCallTools, setPreCallTools] = useState(defaultPreCallTools);
  const [promptVersion, setPromptVersion] = useState(null);
  const [promptRuntimeEnabled, setPromptRuntimeEnabled] = useState(false);
  const [runtimeEvents, setRuntimeEvents] = useState([]);
  const [feedState, setFeedState] = useState(dashboardApiConfigured ? "connecting" : "demo");
  const [savingInstruction, setSavingInstruction] = useState(false);
  const reviewTriggerRef = useRef(null);
  const demo = dashboardDemo;
  const pendingReviews = useMemo(() => demo.reviews.filter((item) => !reviewed.has(item.id)), [reviewed]);

  useEffect(() => {
    if (!dashboardApiConfigured) return;
    fetchPromptPlan()
      .then((plan) => {
        setPromptVersion(plan.version);
        setPromptRuntimeEnabled(Boolean(plan.runtime_enabled));
        if (plan.nodes?.length) setInstructionNodes(nodesFromPlan(plan.nodes));
      })
      .catch((error) => setNotice(`Prompt plan unavailable: ${error.message}`));
  }, []);

  useEffect(() => {
    if (!dashboardApiConfigured) return;
    const controller = new AbortController();
    streamRuntimeEvents((event) => {
      setFeedState("live");
      setRuntimeEvents((items) => [event, ...items].slice(0, 60));
    }, controller.signal).catch((error) => {
      if (!controller.signal.aborted) setFeedState("offline");
      if (!controller.signal.aborted) setNotice(`Live activity paused: ${error.message}`);
    });
    return () => controller.abort();
  }, []);

  const openReview = (item, event) => {
    reviewTriggerRef.current = event.currentTarget;
    setSelectedReview(item);
  };
  const closeReview = () => {
    setSelectedReview(null);
    requestAnimationFrame(() => reviewTriggerRef.current?.focus());
  };
  const markReviewed = (id, label) => {
    setReviewed((existing) => new Set([...existing, id]));
    setReviewedLabels((existing) => ({ ...existing, [id]: label }));
    closeReview();
    setLastReviewed({ id, label });
  };
  const undoReview = () => { if (lastReviewed) { setReviewed((existing) => { const next = new Set(existing); next.delete(lastReviewed.id); return next; }); setReviewedLabels((existing) => { const next = { ...existing }; delete next[lastReviewed.id]; return next; }); } setLastReviewed(null); };
  const addInstruction = async (instruction) => {
    if (!dashboardApiConfigured) {
      setInstructionNodes((nodes) => [...nodes, { id: `local-${Date.now()}`, name: "Custom instruction", detail: "Always · local preview", instruction }]);
      setNotice("Saved in this browser preview. Set VITE_AGENT_API_URL to publish it to calls.");
      return true;
    }
    setSavingInstruction(true);
    try {
      const plan = await appendPromptNode({ instruction });
      setInstructionNodes(nodesFromPlan(plan.nodes));
      setPromptVersion(plan.version);
      setPromptRuntimeEnabled(Boolean(plan.runtime_enabled));
      setNotice(`Published prompt version ${plan.version}. New calls now receive this rule.`);
      return true;
    } catch (error) {
      setNotice(`Could not publish instruction: ${error.message}`);
      return false;
    } finally { setSavingInstruction(false); }
  };

  return (
    <main className="phone-frame">
      <div className="app-shell">
        <div className="app-content">
        {activeTab === "home" && <header className="topbar">
          <div>
            <p className="time-label">Tuesday · 14 October</p>
            <h1>Good evening, Tushar</h1>
          </div>
          <button className="icon-button" aria-label="View notifications" onClick={() => setNotice("No new notifications. Demo data is shown below.")}><Icon name="bell" /></button>
        </header>}

        {activeTab === "home" && <>
          <div className="demo-notice" role="status">{feedState === "live" ? "Live runtime connected · activity updates as calls move." : feedState === "connecting" ? "Connecting to the live runtime…" : "Preview data · Set VITE_AGENT_API_URL to connect calls and prompts."}</div>
          <section className="agent-strip" aria-label="Agent status">
            <div className="agent-mark" aria-hidden="true"><span>R</span></div>
            <div className="agent-copy"><strong>{demo.agent.name}</strong><span>Your on-call assistant</span></div>
            <StatusPill>Ready on device</StatusPill>
          </section>

          <section className="live-call" aria-labelledby="live-call-heading">
            <div className="section-heading"><div><h2 id="live-call-heading">Call in motion</h2></div><button className="quiet-button" onClick={() => setCallLive((value) => !value)}>{callLive ? "Pause demo" : "Resume demo"}</button></div>
            <div className={`call-route ${callLive ? "is-live" : ""}`}>
              <div className="route-stop caller"><span>1</span><small>Caller</small></div>
              <div className="route-line"><b className="signal" /></div>
              <div className="route-stop agent"><span>2</span><small>Relay</small></div>
              <div className="route-line muted"><b /></div>
              <div className="route-stop outcome"><span>3</span><small>Reminder</small></div>
            </div>
            <div className="call-detail"><div><strong>{demo.activeCall.caller}</strong><span>{demo.activeCall.started} · {demo.activeCall.state}</span></div><button className="call-control" onClick={() => setCallLive((value) => !value)} aria-label={callLive ? "Pause demo call" : "Resume demo call"}><Icon name={callLive ? "pause" : "play"} size={17} /></button></div>
            <p className="transcript">“{demo.activeCall.transcript}”</p>
            <div className="action-draft"><div><span>Next action</span><strong>Tomorrow · 09:00</strong></div><p>Reminder draft ready</p><Icon name="check" size={18} /></div>
          </section>

          <section className="today-section" aria-labelledby="today-heading">
            <div className="section-heading"><div><h2 id="today-heading">Your day at a glance</h2></div><button className="text-link" onClick={() => setActiveTab("calls")}>All calls <Icon name="chevron" size={15} /></button></div>
            <div className="metrics-row">
              <div><strong>{demo.today.calls}</strong><span>calls handled</span></div>
              <div><strong>{demo.today.quality}</strong><span>flow quality</span></div>
              <div><strong>{demo.today.review}</strong><span>to review</span></div>
            </div>
          </section>

          <section className="review-section" aria-labelledby="review-heading">
            <div className="section-heading"><div><h2 id="review-heading">One decision needs you</h2></div><button className="text-link" onClick={() => setActiveTab("review")}>Review <Icon name="chevron" size={15} /></button></div>
            {pendingReviews.slice(0, 1).map((review) => <ReviewPreview key={review.id} item={review} onOpen={(event) => openReview(review, event)} />)}
            {!pendingReviews.length && <div className="empty-line"><Icon name="check" /> Everything in tonight’s review queue is handled.</div>}
          </section>
        </>}

        {activeTab === "calls" && <CallsView recent={demo.recent} events={runtimeEvents} feedState={feedState} />}
        {activeTab === "review" && <ReviewView reviews={pendingReviews} onOpen={openReview} />}
        {activeTab === "settings" && <SettingsView
          instructionNodes={instructionNodes}
          setInstructionNodes={setInstructionNodes}
          preCallTools={preCallTools}
          setPreCallTools={setPreCallTools}
          promptVersion={promptVersion}
          promptRuntimeEnabled={promptRuntimeEnabled}
          isLive={dashboardApiConfigured}
          savingInstruction={savingInstruction}
          onAddInstruction={addInstruction}
          onNotice={setNotice}
        />}

        </div>

        <nav className="bottom-nav" aria-label="Primary navigation">
          {[['home','Home'],['calls','Calls'],['review','Review'],['settings','Setup']].map(([id, label]) => <button key={id} className={activeTab === id ? "active" : ""} onClick={() => setActiveTab(id)} aria-current={activeTab === id ? "page" : undefined}><Icon name={id} /><span>{label}</span>{id === "review" && pendingReviews.length > 0 && <b className="nav-count">{pendingReviews.length}</b>}</button>)}
        </nav>

        {selectedReview && <ReviewSheet item={selectedReview} onClose={closeReview} onApprove={(label) => markReviewed(selectedReview.id, label)} />}
        {lastReviewed && <div className="undo-toast" role="status"><span>Saved as {lastReviewed.label.replace('_', ' ')}</span><button onClick={undoReview}>Undo</button></div>}
        {notice && <div className="notice-toast" role="status"><span>{notice}</span><button onClick={() => setNotice("")}>Dismiss</button></div>}
      </div>
    </main>
  );
}

function ReviewPreview({ item, onOpen }) { return <button className="review-preview" onClick={onOpen}><div><span className="intent-label">Candidate · {item.intent.replace('_', ' ')}</span><p>{item.text}</p></div><div className="confidence"><strong>{item.confidence}</strong><span>confidence</span></div><Icon name="chevron" size={17} /></button>; }
function ReviewView({ reviews, onOpen }) { return <section className="tab-page"><h2>Candidate labels</h2><p className="page-intro">These are MiniLM suggestions. A review never retrains the model automatically.</p>{reviews.length ? <div className="review-list">{reviews.map((item) => <ReviewPreview key={item.id} item={item} onOpen={(event) => onOpen(item, event)} />)}</div> : <div className="empty-state"><Icon name="check" size={24} /><h3>Queue cleared</h3><p>Your next nightly batch will appear here after classification.</p></div>}</section>; }
function CallsView({ recent, events, feedState }) { return <section className="tab-page activity-page"><div className="activity-heading"><div><h2>Live activity</h2><p className="page-intro">Transcript, response, and shadow-classifier events from this running API process.</p></div><StatusPill tone={feedState === "live" ? "mint" : "slate"}>{feedState === "live" ? "Live feed" : feedState === "connecting" ? "Connecting" : "Preview"}</StatusPill></div>{events.length ? <div className="event-list" aria-live="polite">{events.map((event, index) => <article className="event-row" key={`${event.at}-${index}`}><span className={`event-dot ${event.kind.split(".")[0]}`} /><div><strong>{runtimeEventTitle(event)}</strong><p>{runtimeEventDetail(event)}</p></div><time>{new Date(event.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></article>)}</div> : <div className="event-empty"><Icon name="refresh" size={21} /><strong>{feedState === "connecting" ? "Waiting for the runtime" : "No live events yet"}</strong><p>Start a call or submit a turn; its transcript and MiniLM shadow result will appear here.</p></div>}<div className="activity-divider" /><h3 className="activity-subhead">Recent calls</h3><div className="call-list">{recent.map((call) => <article key={call.time} className="call-row"><div className="avatar">{call.name[0]}</div><div><strong>{call.name}</strong><span>{call.topic}</span></div><time>{call.time}</time><StatusPill tone={call.outcome === 'Resolved' ? 'mint' : 'amber'}>{call.outcome}</StatusPill></article>)}</div></section>; }
function SettingsView({ instructionNodes, setInstructionNodes, preCallTools, setPreCallTools, promptVersion, promptRuntimeEnabled, isLive, savingInstruction, onAddInstruction, onNotice }) {
  const [section, setSection] = useState("instructions");
  const [showEditor, setShowEditor] = useState(false);
  const [newInstruction, setNewInstruction] = useState("");
  const [showServer, setShowServer] = useState(false);
  const [serverName, setServerName] = useState("");
  const enabledCount = preCallTools.filter((tool) => tool.enabled).length;
  const addInstruction = async () => {
    const instruction = newInstruction.trim();
    if (!instruction) return;
    if (await onAddInstruction(instruction)) { setNewInstruction(""); setShowEditor(false); }
  };
  const addServer = () => {
    if (!serverName.trim()) return;
    setShowServer(false); onNotice(`${serverName.trim()} was added as a local preview. Secrets are never stored in the browser.`); setServerName("");
  };
  const toggleTool = (id) => setPreCallTools((tools) => tools.map((tool) => tool.id === id ? { ...tool, enabled: !tool.enabled } : tool));
  return <section className="tab-page setup-page">
    <div className="setup-head"><div><h2>Agent setup</h2><p className="page-intro">Shape what Relay knows before a caller says hello.</p></div><StatusPill tone={isLive && promptRuntimeEnabled ? "mint" : "slate"}>{isLive && promptRuntimeEnabled ? `Published${promptVersion ? ` v${promptVersion}` : ""}` : isLive ? "Stored · off" : "Preview"}</StatusPill></div>
    <div className="setup-tabs" role="tablist" aria-label="Agent setup">
      <button role="tab" aria-selected={section === "instructions"} className={section === "instructions" ? "selected" : ""} onClick={() => setSection("instructions")}><Icon name="branch" size={17} />Instructions</button>
      <button role="tab" aria-selected={section === "mcp"} className={section === "mcp" ? "selected" : ""} onClick={() => setSection("mcp")}><Icon name="plug" size={17} />MCP prep</button>
    </div>
    {section === "instructions" ? <>
      <section className="setup-hero" aria-labelledby="instruction-stack"><div className="setup-hero-mark"><Icon name="sliders" size={21} /></div><div><h3 id="instruction-stack">Instruction stack</h3><p>{instructionNodes.length} small rules assemble into the active voice prompt</p></div><span>{promptVersion ? `v${promptVersion}` : "local"}</span></section>
      <div className="setup-route" aria-label="Prompt assembly"><span>Rules</span><i /><span>Prompt</span><i /><span>Call</span></div>
      <div className="instruction-list">{instructionNodes.map((node, index) => <article className="instruction-row" key={node.id}><div className="instruction-order">{index + 1}</div><div><strong>{node.name}</strong><small>{node.detail}</small><p>{node.instruction}</p></div><button className="row-icon" aria-label={`View ${node.name}`} onClick={() => onNotice(node.instruction)}><Icon name="eye" size={17} /></button></article>)}</div>
      {showEditor ? <div className="inline-editor"><label htmlFor="instruction">New instruction</label><textarea id="instruction" value={newInstruction} onChange={(event) => setNewInstruction(event.target.value)} placeholder="Tell Relay exactly how to respond in this situation" /><div><button className="change-button" onClick={() => setShowEditor(false)} disabled={savingInstruction}>Cancel</button><button className="compact-primary" onClick={addInstruction} disabled={savingInstruction}>{savingInstruction ? "Publishing…" : "Add rule"}</button></div></div> : <button className="outline-action" onClick={() => setShowEditor(true)}><Icon name="plus" size={18} />Add instruction</button>}
    </> : <>
      <section className="setup-hero mcp-hero" aria-labelledby="pre-call-readiness"><div className="setup-hero-mark"><Icon name="database" size={21} /></div><div><h3 id="pre-call-readiness">Pre-call readiness</h3><p>{enabledCount} read-only source{enabledCount === 1 ? "" : "s"} prepare caller context</p></div><span>Ready</span></section>
      <p className="safe-note"><Icon name="shield" size={16} />Only approved read-only tools run before a call. The browser never receives credentials.</p>
      <section className="mcp-connection"><div className="connection-top"><div><span>Remote MCP server</span><strong>Customer context gateway</strong><small>Streamable HTTP · token stored server-side</small></div><StatusPill>Connected</StatusPill></div><button className="plain-link" onClick={() => onNotice("Tool discovery will run from the backend after the MCP configuration API is connected.")}>Refresh tool list <Icon name="refresh" size={16} /></button></section>
      <div className="tool-list">{preCallTools.map((tool) => <article className="tool-row" key={tool.id}><div className="tool-icon"><Icon name={tool.icon} size={18} /></div><div><strong>{tool.name}</strong><small>{tool.description}</small><em>{tool.scope}</em></div><button className={`tool-toggle ${tool.enabled ? "on" : ""}`} aria-pressed={tool.enabled} aria-label={`${tool.enabled ? "Disable" : "Enable"} ${tool.name} before calls`} onClick={() => toggleTool(tool.id)}><i /></button></article>)}</div>
      {showServer ? <div className="inline-editor"><label htmlFor="server">Server name</label><input id="server" value={serverName} onChange={(event) => setServerName(event.target.value)} placeholder="e.g. CRM context" /><small>Endpoint and secret reference are saved by the backend, not this WebView.</small><div><button className="change-button" onClick={() => setShowServer(false)}>Cancel</button><button className="compact-primary" onClick={addServer}>Save server</button></div></div> : <button className="outline-action" onClick={() => setShowServer(true)}><Icon name="plus" size={18} />Add MCP server</button>}
    </>}
  </section>;
}

function nodesFromPlan(nodes) {
  return [...nodes]
    .sort((left, right) => left.priority - right.priority)
    .map((node) => ({
      id: node.id,
      name: node.condition?.language ? `${node.condition.language} rule` : "Prompt instruction",
      detail: `${Object.keys(node.condition || {}).length ? Object.entries(node.condition).map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join("/") : value}`).join(" · ") : "Always"} · priority ${node.priority}`,
      instruction: node.instruction,
    }));
}

function runtimeEventTitle(event) {
  const titles = {
    "call.started": "Call opened",
    "call.ended": "Call closed",
    "transcript.received": "Caller transcript received",
    "classifier.queued": "MiniLM shadow pass queued",
    "classifier.completed": "MiniLM candidate ready",
    "classifier.failed": "MiniLM shadow pass failed",
    "agent.responded": "Relay response ready",
    "prompt.published": "Prompt plan published",
  };
  return titles[event.kind] || event.kind.replace(".", " · ");
}

function runtimeEventDetail(event) {
  if (event.kind === "transcript.received") return event.text;
  if (event.kind === "agent.responded") return event.text;
  if (event.kind === "classifier.completed") return `${String(event.label || "unknown").replace("_", " ")} · ${event.confidence == null ? "confidence unavailable" : `${Math.round(event.confidence * 100)}% confidence`}`;
  if (event.kind === "classifier.failed") return event.message || "Check the model path and retail-classifier dependency on the API host.";
  if (event.kind === "classifier.queued") return "Running beside the generic path; it cannot delay the reply.";
  if (event.kind === "prompt.published") return `Version ${event.version} now contains ${event.nodes} instruction${event.nodes === 1 ? "" : "s"}.`;
  return event.session_id ? `Session ${event.session_id}` : "Runtime event";
}

const defaultInstructionNodes = [
  { id: "identity", name: "Identity disclosure", detail: "Always · priority 10", instruction: "Tell callers you are the owner's AI assistant on the first reply." },
  { id: "privacy", name: "Privacy boundary", detail: "Always · priority 20", instruction: "Keep personal data, OTPs, messages, and financial details private." },
  { id: "voice", name: "Voice-call behavior", detail: "Voice · priority 40", instruction: "Ask one focused follow-up only when it is needed to act safely." },
  { id: "language", name: "Reply language", detail: "Hinglish · priority 50", instruction: "Reply in natural Roman-script Hinglish unless the caller uses another language." },
];

const defaultPreCallTools = [
  { id: "profile", name: "Lookup caller profile", description: "Known name and relationship", scope: "Read-only · before every call", icon: "database", enabled: true },
  { id: "orders", name: "Recent orders", description: "Open orders and delivery state", scope: "Read-only · before every call", icon: "link", enabled: true },
  { id: "reminders", name: "Open reminders", description: "Pending callbacks and timing", scope: "On demand during the call", icon: "bell", enabled: false },
];
function ReviewSheet({ item, onClose, onApprove }) { const [label, setLabel] = useState(item.intent); const [changing, setChanging] = useState(false); const closeRef = useRef(null); const sheetRef = useRef(null); useEffect(() => { closeRef.current?.focus(); const onKey = (event) => { if (event.key === "Escape") onClose(); }; window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey); }, [onClose]); const trapFocus = (event) => { if (event.key !== "Tab") return; const focusable = [...sheetRef.current.querySelectorAll("button, select")].filter((element) => !element.disabled); const first = focusable[0]; const last = focusable[focusable.length - 1]; if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); } else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); } }; const approve = () => onApprove(label); return <div className="sheet-backdrop" role="presentation" onMouseDown={onClose}><section ref={sheetRef} className="review-sheet" role="dialog" aria-modal="true" aria-labelledby="review-sheet-title" onKeyDown={trapFocus} onMouseDown={(event) => event.stopPropagation()}><div className="sheet-grabber" /><button ref={closeRef} className="sheet-close" onClick={onClose}>Close</button><p className="review-meta">{item.id} · {item.time}</p><h2 id="review-sheet-title">Is this intent right?</h2><blockquote>{item.text}</blockquote><div className="candidate-answer"><span>MiniLM suggests</span><strong>{item.intent.replace('_', ' ')}</strong><small>{item.confidence} confidence · candidate only</small></div>{changing && <label className="label-picker"><span>Final label</span><select value={label} onChange={(event) => setLabel(event.target.value)}>{["track_order","return_item","cancel_order","refund_status","wrong_item","size_issue","payment_issue","delivery_address","other"].map((option) => <option key={option} value={option}>{option.replace('_', ' ')}</option>)}</select></label>}<button className="approve-button" onClick={approve}><Icon name="check" /> {changing ? "Save correction" : "Yes, keep this label"}</button><button className="change-button" onClick={() => setChanging((value) => !value)}>{changing ? "Keep suggested label" : "Change label"}</button></section></div>; }

export default App;
