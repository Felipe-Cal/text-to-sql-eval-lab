import React, { useState, useRef, useEffect, useCallback } from 'react';
import './index.css';

const API_BASE = 'http://127.0.0.1:8000';

const STRATEGIES = [
  { group: 'Standard',  id: 'zero_shot',        label: 'Zero Shot' },
  { group: 'Standard',  id: 'few_shot_static',   label: 'Few Shot Static' },
  { group: 'Standard',  id: 'few_shot_dynamic',  label: 'Few Shot Dynamic' },
  { group: 'Standard',  id: 'chain_of_thought',  label: 'Chain of Thought' },
  { group: 'RAG',       id: 'rag_dense',         label: 'RAG Dense' },
  { group: 'RAG',       id: 'rag_sparse',        label: 'RAG Sparse' },
  { group: 'RAG',       id: 'rag_hybrid',        label: 'RAG Hybrid' },
  { group: 'Advanced',  id: 'tool_use',          label: 'Tool Use' },
  { group: 'Advanced',  id: 'dspy',              label: 'DSPy Optimized' },
  { group: 'Advanced',  id: 'routed',            label: 'Smart Router' },
];

const MODELS = [
  { group: 'OpenAI',    id: 'openai/gpt-4o',              label: 'GPT-4o' },
  { group: 'OpenAI',    id: 'openai/gpt-4o-mini',         label: 'GPT-4o Mini' },
  { group: 'Anthropic', id: 'anthropic/claude-sonnet-4-6',         label: 'Claude Sonnet 4.6' },
  { group: 'Anthropic', id: 'anthropic/claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5' },
  { group: 'Local',     id: 'meta-llama/Meta-Llama-3.1-8B-Instruct', label: 'Llama 3.1 8B (vLLM)' },
];

const DIFFICULTIES = ['easy', 'medium', 'hard'];

const TOOL_ICONS = {
  query_database:        '🗄️',
  search_knowledge_base: '📚',
  get_schema:            '🔍',
};

// ─── Sub-components ──────────────────────────────────────────────────────────

function BackendBadge({ health, selectedModel }) {
  if (!health) {
    return (
      <div className="backend-badge warn">
        <span className="backend-dot" />
        Connecting…
      </div>
    );
  }
  const isOk = health.status === 'ok';
  // Derive a short display name from the selected model
  const modelEntry = MODELS.find(m => m.id === selectedModel);
  const label = modelEntry ? modelEntry.label : (selectedModel || '').split('/').pop();
  return (
    <div className={`backend-badge ${isOk ? 'ok' : 'err'}`}>
      <span className="backend-dot" />
      {label}
    </div>
  );
}

function ToolCallItem({ call }) {
  const [expanded, setExpanded] = useState(false);
  const icon = TOOL_ICONS[call.tool] || '🔧';
  const statusClass = call.pending ? 'pending' : call.success ? 'success' : 'error';
  const statusLabel = call.pending ? 'Running…' : call.success ? '✓ Done' : '✗ Failed';

  return (
    <div className={`tool-call-item ${statusClass}`}>
      <div className="tool-call-header" onClick={() => !call.pending && setExpanded(e => !e)}>
        <span className="tool-icon">{icon}</span>
        <span className="tool-name">{call.tool}</span>
        <span className={`tool-status ${statusClass}`}>{statusLabel}</span>
        {!call.pending && <span className="expand-icon">{expanded ? '▾' : '▸'}</span>}
      </div>
      {expanded && (
        <div className="tool-call-body">
          {call.input != null && (
            <div className="tool-io">
              <span className="tool-io-label">Input</span>
              <pre>{typeof call.input === 'string' ? call.input : JSON.stringify(call.input, null, 2)}</pre>
            </div>
          )}
          {call.output != null && (
            <div className="tool-io">
              <span className="tool-io-label">Output</span>
              <pre>{call.output}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SqlBlock({ sql, isStreaming }) {
  const [copied, setCopied] = useState(false);
  if (!sql && !isStreaming) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(sql || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <div className="sql-block">
      <div className="sql-block-header">
        <span className="sql-block-label">SQL</span>
        {!isStreaming && sql && (
          <button className={`copy-btn ${copied ? 'copied' : ''}`} onClick={handleCopy}>
            {copied ? 'Copied!' : 'Copy'}
          </button>
        )}
      </div>
      <pre className="sql-code">
        {sql || ''}
        {isStreaming && <span className="sql-cursor" />}
      </pre>
    </div>
  );
}

function DataTable({ rows, error }) {
  if (error) return <p className="ai-error">Query error: {error}</p>;
  if (!rows || rows.length === 0) return (
    <p style={{ fontSize: '0.83em', color: 'var(--text-3)', fontStyle: 'italic', marginBottom: 12 }}>
      No rows returned.
    </p>
  );
  const headers = Object.keys(rows[0]);
  return (
    <div className="data-wrap">
      <table className="data-table">
        <thead>
          <tr>{headers.map(h => <th key={h}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {headers.map(h => <td key={h}>{String(row[h] ?? '')}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricsBadges({ metrics, toolCount }) {
  if (!metrics) return null;
  return (
    <div className="metrics-row">
      {metrics.latency != null && (
        <span className="badge">{metrics.latency.toFixed(2)}s</span>
      )}
      {metrics.cost > 0 && (
        <span className="badge">${metrics.cost.toFixed(5)}</span>
      )}
      {metrics.tokens > 0 && (
        <span className="badge">{metrics.tokens} tokens</span>
      )}
      {metrics.attempts > 1 && (
        <span className="badge yellow">{metrics.attempts} attempts</span>
      )}
      {toolCount > 0 && (
        <span className="badge blue">{toolCount} tool call{toolCount !== 1 ? 's' : ''}</span>
      )}
      {metrics.routedDifficulty && (
        <span className="badge accent">
          {metrics.routedDifficulty} · {metrics.routerMethod}
        </span>
      )}
    </div>
  );
}

function AiMessage({ msg }) {
  const { isStreaming, streamingSql, sql, answer, toolEvents, reasoning, retryCount, data, dataError, metrics, text, error } = msg;

  const displaySql = isStreaming ? streamingSql : sql;
  const toolCount = toolEvents?.length || 0;

  return (
    <div className="ai-row">
      <div className="ai-avatar">✦</div>
      <div className="ai-body">
        {error && <div className="ai-error">{error}</div>}
        {text && <div className="ai-text">{text}</div>}
        {answer && <div className="ai-answer">{answer}</div>}
        {reasoning && <div className="reasoning-box">{reasoning}</div>}
        {retryCount > 0 && (
          <div className="retry-notice">
            ↺ Retrying after SQL error ({retryCount} attempt{retryCount !== 1 ? 's' : ''})
          </div>
        )}
        {toolEvents?.length > 0 && (
          <div className="tool-timeline">
            {toolEvents.map((tc, i) => <ToolCallItem key={i} call={tc} />)}
          </div>
        )}
        {(displaySql || isStreaming) && (
          <SqlBlock sql={displaySql} isStreaming={isStreaming} />
        )}
        {data !== null && data !== undefined && (
          <DataTable rows={data} error={dataError} />
        )}
        {!isStreaming && (
          <MetricsBadges metrics={metrics} toolCount={toolCount} />
        )}
      </div>
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');

  // ── Chat state
  const [messages, setMessages] = useState([{
    role: 'ai',
    text: 'Hello! I\'m your Text-to-SQL assistant, connected to the e-commerce DuckDB sandbox.\n\nTry asking:\n• "What is total revenue per product category?"\n• "Which customers placed the most orders?"\n• "What is our return rate?" (try Tool Use for policy questions)',
  }]);
  const [input, setInput] = useState('');
  const [chatStrategy, setChatStrategy] = useState('few_shot_dynamic');
  const [chatModel, setChatModel] = useState('openai/gpt-4o-mini');
  const [isStreaming, setIsStreaming] = useState(false);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  // ── Backend health
  const [backendHealth, setBackendHealth] = useState(null);

  // ── Eval state
  const [selectedDifficulties, setSelectedDifficulties] = useState(['hard']);
  const [selectedStrategies, setSelectedStrategies] = useState(['rag_hybrid', 'few_shot_dynamic']);
  const [evalModel, setEvalModel] = useState('openai/gpt-4o-mini');
  const [expandedRows, setExpandedRows] = useState([]);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'desc' });
  const activePolls = useRef({});

  const [jobs, setJobs] = useState(() => {
    const saved = localStorage.getItem('sqlEvalJobsCache_v2');
    if (saved) { try { return JSON.parse(saved); } catch { return []; } }
    return [];
  });

  // Persist completed jobs
  useEffect(() => {
    const done = jobs.filter(j => ['completed', 'failed'].includes(j.status));
    localStorage.setItem('sqlEvalJobsCache_v2', JSON.stringify(done));
  }, [jobs]);

  // Fetch backend health on mount
  useEffect(() => {
    fetch(`${API_BASE}/health/backend`)
      .then(r => r.json())
      .then(setBackendHealth)
      .catch(() => setBackendHealth({ status: 'unavailable', backend: 'unknown' }));
  }, []);

  // Auto-scroll
  useEffect(() => {
    if (activeTab === 'chat' && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, activeTab]);

  // Helper: update the last (streaming) AI message
  const updateLast = useCallback((updater) => {
    setMessages(prev => {
      if (!prev.length) return prev;
      const last = prev[prev.length - 1];
      if (last.role !== 'ai') return prev;
      return [...prev.slice(0, -1), { ...last, ...updater(last) }];
    });
  }, []);

  // Fetch SQL results after streaming completes
  const fetchResults = useCallback(async (sql) => {
    try {
      const res = await fetch(`${API_BASE}/sql/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sql }),
      });
      const { data, error } = await res.json();
      setMessages(prev => {
        // Find the last AI message that has this SQL
        for (let i = prev.length - 1; i >= 0; i--) {
          if (prev[i].role === 'ai' && prev[i].sql === sql) {
            return prev.map((m, idx) =>
              idx === i ? { ...m, data: data ?? [], dataError: error } : m
            );
          }
        }
        return prev;
      });
    } catch { /* skip data on network error */ }
  }, []);

  const handleSubmitChat = async (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || isStreaming) return;

    setInput('');
    setIsStreaming(true);

    setMessages(prev => [
      ...prev,
      { role: 'user', text: question },
      {
        role: 'ai', isStreaming: true,
        streamingSql: '', sql: '', answer: '',
        toolEvents: [], reasoning: null,
        retryCount: 0, data: null, dataError: null,
        metrics: null, text: null, error: null,
      },
    ]);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const resp = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, strategy: chatStrategy, model: chatModel }),
        signal: ctrl.signal,
      });

      if (!resp.ok) throw new Error(`Server error ${resp.status}`);

      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          try {
            const ev = JSON.parse(raw);
            switch (ev.type) {
              case 'sql_token':
                updateLast(m => ({ streamingSql: (m.streamingSql || '') + ev.content }));
                break;
              case 'tool_call':
                updateLast(m => ({
                  toolEvents: [...(m.toolEvents || []), {
                    tool: ev.tool, input: ev.input,
                    output: null, success: null, pending: true,
                  }],
                }));
                break;
              case 'tool_result':
                updateLast(m => {
                  const evts = [...(m.toolEvents || [])];
                  for (let i = evts.length - 1; i >= 0; i--) {
                    if (evts[i].tool === ev.tool && evts[i].pending) {
                      evts[i] = { ...evts[i], output: ev.output, success: ev.success, pending: false };
                      break;
                    }
                  }
                  return { toolEvents: evts };
                });
                break;
              case 'retry':
                updateLast(m => ({ streamingSql: '', retryCount: (m.retryCount || 0) + 1 }));
                break;
              case 'done':
                updateLast(() => ({
                  isStreaming: false,
                  sql: ev.sql || '',
                  streamingSql: '',
                  answer: ev.answer || '',
                  reasoning: ev.reasoning || null,
                  metrics: {
                    latency: ev.latency,
                    cost: ev.cost,
                    tokens: (ev.prompt_tokens || 0) + (ev.completion_tokens || 0),
                    attempts: ev.attempts,
                    routedDifficulty: ev.routed_difficulty,
                    routerMethod: ev.router_method,
                  },
                }));
                if (ev.sql) fetchResults(ev.sql);
                break;
              case 'error':
                updateLast(() => ({ isStreaming: false, error: ev.message }));
                break;
            }
          } catch { /* skip malformed event */ }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        updateLast(() => ({ isStreaming: false, error: err.message }));
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
      // Ensure streaming flag is cleared even if done event was missed
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.isStreaming) {
          return [...prev.slice(0, -1), { ...last, isStreaming: false }];
        }
        return prev;
      });
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmitChat(e);
    }
  };

  // ── Eval logic ────────────────────────────────────────────────────────────

  const isAnyJobRunning = jobs.some(j => ['running', 'pending'].includes(j.status));

  const toggleStrategy = id =>
    setSelectedStrategies(p => p.includes(id) ? p.filter(s => s !== id) : [...p, id]);

  const toggleDifficulty = d =>
    setSelectedDifficulties(p => p.includes(d) ? p.filter(x => x !== d) : [...p, d]);

  const startBatchEval = async () => {
    if (!selectedStrategies.length || !selectedDifficulties.length) return;
    setExpandedRows(selectedStrategies);

    const combos = selectedStrategies.flatMap(s =>
      selectedDifficulties.map(d => ({ strat: s, diff: d }))
    );

    const kept = [...jobs];
    const toRun = [];

    combos.forEach(({ strat, diff }) => {
      const idx = kept.findIndex(j => j.strategy === strat && j.difficulty === diff);
      if (idx !== -1 && kept[idx].status === 'completed') return;
      toRun.push({ strat, diff });
      if (idx !== -1) kept[idx] = { ...kept[idx], status: 'pending', jobId: null, results: null };
      else kept.push({ jobId: null, strategy: strat, difficulty: diff, status: 'pending', results: null });
    });

    setJobs(kept);
    if (!toRun.length) return;

    const dispatched = await Promise.all(toRun.map(async ({ strat, diff }) => {
      try {
        const res = await fetch(`${API_BASE}/evals/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ strategy: strat, difficulty: diff, model: evalModel }),
        });
        const data = await res.json();
        return { strat, diff, jobId: data.job_id, ok: true };
      } catch {
        return { strat, diff, jobId: null, ok: false };
      }
    }));

    setJobs(cur => cur.map(job => {
      const d = dispatched.find(x => x.strat === job.strategy && x.diff === job.difficulty);
      if (!d) return job;
      if (d.ok && d.jobId) {
        pollJob(d.jobId);
        return { ...job, jobId: d.jobId, status: 'running' };
      }
      return { ...job, status: 'failed' };
    }));
  };

  const pollJob = (jobId) => {
    const id = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/evals/${jobId}`);
        const data = await res.json();
        if (['completed', 'failed'].includes(data.status)) {
          clearInterval(id);
          delete activePolls.current[jobId];
          setJobs(cur => cur.map(j =>
            j.jobId === jobId
              ? { ...j, status: data.status, results: data.results?.scores ?? null }
              : j
          ));
        }
      } catch {
        clearInterval(id);
        delete activePolls.current[jobId];
        setJobs(cur => cur.map(j => j.jobId === jobId ? { ...j, status: 'failed' } : j));
      }
    }, 3000);
    activePolls.current[jobId] = id;
  };

  const clearCache = () => {
    setJobs([]);
    localStorage.removeItem('sqlEvalJobsCache_v2');
  };

  // ── Leaderboard data ──────────────────────────────────────────────────────

  const getVal = (results, path) => {
    if (!results) return null;
    const v = path.split('.').reduce((o, k) => (o ? o[k] : null), results);
    return typeof v === 'number' ? v : null;
  };

  const groupedJobs = STRATEGIES
    .filter(s => selectedStrategies.includes(s.id))
    .map(st => {
      const gJobs = jobs.filter(j => j.strategy === st.id);
      const done = gJobs.filter(j => j.status === 'completed' && j.results);
      const running = gJobs.some(j => ['pending', 'running'].includes(j.status));
      const failed = gJobs.some(j => j.status === 'failed');
      const status = running ? 'running' : failed ? 'failed' : done.length > 0 ? 'done' : 'pending';

      const avg = path => {
        const vals = done.map(j => getVal(j.results, path)).filter(v => v !== null);
        return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
      };

      return {
        id: st.id,
        label: st.label,
        group: st.group,
        status,
        totalJobs: gJobs.length,
        children: gJobs,
        metrics: {
          result_match:    avg('result_match.accuracy'),
          semantic_judge:  avg('semantic_judge.accuracy'),
          retrieval_recall:avg('retrieval_recall.mean'),
          avg_latency:     avg('avg_latency.mean'),
          avg_cost:        avg('avg_cost.mean'),
          avg_attempts:    avg('avg_attempts.mean'),
          avg_tool_calls:  avg('avg_tool_calls.mean'),
        },
      };
    })
    .filter(g => g.totalJobs > 0);

  const handleSort = key =>
    setSortConfig(p => ({ key, direction: p.key === key && p.direction === 'desc' ? 'asc' : 'desc' }));

  const sortedGroups = [...groupedJobs].sort((a, b) => {
    if (!sortConfig.key) return 0;
    if (sortConfig.key === 'strategy') {
      const r = a.label.localeCompare(b.label);
      return sortConfig.direction === 'asc' ? r : -r;
    }
    const va = a.metrics[sortConfig.key], vb = b.metrics[sortConfig.key];
    if (va === null && vb === null) return 0;
    if (va === null) return 1;
    if (vb === null) return -1;
    return sortConfig.direction === 'asc' ? va - vb : vb - va;
  });

  const sortIcon = key => sortConfig.key !== key ? '' : sortConfig.direction === 'asc' ? ' ▴' : ' ▾';

  const fmtVal = (val, key) => {
    if (val === null || val === undefined) return '—';
    if (key.includes('accuracy') || key.includes('recall')) return (val * 100).toFixed(1) + '%';
    if (key.includes('latency')) return val.toFixed(2) + 's';
    if (key.includes('cost')) return '$' + val.toFixed(5);
    if (key.includes('attempts') || key.includes('tool')) return val.toFixed(1);
    return String(val);
  };

  const Cell = ({ status, val, metricKey }) => {
    if (['running', 'pending'].includes(status)) return <span style={{ color: 'var(--text-3)' }}>…</span>;
    if (status === 'failed') return <span style={{ color: 'var(--red)', fontSize: '0.8em' }}>Error</span>;
    return <span>{fmtVal(val, metricKey)}</span>;
  };

  // ── Strategy groups for select ──────────────────
  const strategyGroups = [...new Set(STRATEGIES.map(s => s.group))].map(g => ({
    group: g,
    items: STRATEGIES.filter(s => s.group === g),
  }));

  const modelGroups = [...new Set(MODELS.map(m => m.group))].map(g => ({
    group: g,
    items: MODELS.filter(m => m.group === g),
  }));

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="app-container">

      {/* ── Sidebar ── */}
      <div className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">⚡</div>
            <div>
              <div className="sidebar-title">Text-to-SQL Lab</div>
              <div className="sidebar-subtitle">Evaluation & Benchmarking</div>
            </div>
          </div>
          <BackendBadge health={backendHealth} selectedModel={activeTab === 'chat' ? chatModel : evalModel} />
        </div>

        <div className="tabs" style={{ margin: '12px 12px 0' }}>
          <div className={`tab ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
            Chat
          </div>
          <div className={`tab ${activeTab === 'eval' ? 'active' : ''}`} onClick={() => setActiveTab('eval')}>
            Leaderboard
          </div>
        </div>

        {activeTab === 'chat' && (
          <div className="sidebar-section" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div>
              <div className="section-label">Model</div>
              <select
                className="strategy-select"
                value={chatModel}
                onChange={e => setChatModel(e.target.value)}
              >
                {modelGroups.map(({ group, items }) => (
                  <optgroup key={group} label={group}>
                    {items.map(m => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>

            <div>
              <div className="section-label">Strategy</div>
              <select
                className="strategy-select"
                value={chatStrategy}
                onChange={e => setChatStrategy(e.target.value)}
              >
                {strategyGroups.map(({ group, items }) => (
                  <optgroup key={group} label={group}>
                    {items.map(s => (
                      <option key={s.id} value={s.id}>{s.label}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
          </div>
        )}

        {activeTab === 'eval' && (
          <>
            <div className="sidebar-section">
              <div className="section-label">Model</div>
              <select
                className="strategy-select"
                value={evalModel}
                onChange={e => setEvalModel(e.target.value)}
                disabled={isAnyJobRunning}
              >
                {modelGroups.map(({ group, items }) => (
                  <optgroup key={group} label={group}>
                    {items.map(m => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>

            <div className="sidebar-section">
              <div className="section-label">Dataset Difficulty</div>
              <div className="check-list">
                {DIFFICULTIES.map(d => (
                  <label key={d} className={`check-item ${isAnyJobRunning ? 'disabled' : ''}`}>
                    <input
                      type="checkbox"
                      checked={selectedDifficulties.includes(d)}
                      onChange={() => toggleDifficulty(d)}
                    />
                    {d.charAt(0).toUpperCase() + d.slice(1)}
                  </label>
                ))}
              </div>
            </div>

            <div className="sidebar-section" style={{ flex: 1 }}>
              <div className="section-label">Compare Strategies</div>
              <div className="check-list">
                {STRATEGIES.map(s => (
                  <label key={s.id} className={`check-item ${isAnyJobRunning ? 'disabled' : ''}`}>
                    <input
                      type="checkbox"
                      checked={selectedStrategies.includes(s.id)}
                      onChange={() => toggleStrategy(s.id)}
                    />
                    {s.label}
                  </label>
                ))}
              </div>
            </div>

            <div className="sidebar-section">
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="btn-primary"
                  style={{ flex: 1 }}
                  onClick={startBatchEval}
                  disabled={isAnyJobRunning || !selectedStrategies.length || !selectedDifficulties.length}
                >
                  {isAnyJobRunning ? 'Running…' : 'Run Evals'}
                </button>
                <button className="btn-danger" onClick={clearCache} title="Clear results cache">
                  🗑
                </button>
              </div>
            </div>
          </>
        )}

        <div className="sidebar-footer">
          {(() => {
            const m = activeTab === 'chat' ? chatModel : evalModel;
            const entry = MODELS.find(x => x.id === m);
            return `Inference: ${entry?.group || 'API'} · ${entry?.label || m}`;
          })()}
        </div>
      </div>

      {/* ── Main area ── */}
      {activeTab === 'chat' ? (

        <div className="chat-container">
          <div className="messages" ref={scrollRef}>
            {messages.map((msg, i) => (
              <div key={i} className={`message-row ${msg.role === 'user' ? 'user-row' : ''}`}>
                {msg.role === 'user'
                  ? <div className="user-bubble">{msg.text}</div>
                  : <AiMessage msg={msg} />
                }
              </div>
            ))}
            {isStreaming && (
              <div className="loading-row">
                <div className="loading-dots">
                  <span /><span /><span />
                </div>
              </div>
            )}
          </div>

          <div className="input-area">
            <form className="input-form" onSubmit={handleSubmitChat}>
              <textarea
                ref={inputRef}
                className="input-field"
                placeholder="Ask about the data…"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isStreaming}
                rows={1}
              />
              <button className="send-btn" type="submit" disabled={isStreaming || !input.trim()}>
                ↑
              </button>
            </form>
          </div>
        </div>

      ) : (

        <div className="eval-dashboard">
          <div className="eval-header">
            <h1 className="eval-title">Evaluation Leaderboard</h1>
            {isAnyJobRunning && <div className="spinner" />}
          </div>
          <p className="eval-description">
            Select strategies and difficulties on the left. Cached results reload instantly. Click column headers to sort.
          </p>

          {sortedGroups.length > 0 ? (
            <div className="leaderboard-wrap">
              <table className="leaderboard">
                <thead>
                  <tr>
                    <th onClick={() => handleSort('strategy')}>Strategy{sortIcon('strategy')}</th>
                    <th>Status</th>
                    <th onClick={() => handleSort('result_match')}>Result Match{sortIcon('result_match')}</th>
                    <th onClick={() => handleSort('semantic_judge')}>Semantic{sortIcon('semantic_judge')}</th>
                    <th onClick={() => handleSort('retrieval_recall')}>RAG Recall{sortIcon('retrieval_recall')}</th>
                    <th onClick={() => handleSort('avg_attempts')}>Attempts{sortIcon('avg_attempts')}</th>
                    <th onClick={() => handleSort('avg_tool_calls')}>Tool Calls{sortIcon('avg_tool_calls')}</th>
                    <th onClick={() => handleSort('avg_latency')}>Latency{sortIcon('avg_latency')}</th>
                    <th onClick={() => handleSort('avg_cost')}>Cost{sortIcon('avg_cost')}</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedGroups.map(group => {
                    const isExpanded = expandedRows.includes(group.id);
                    return (
                      <React.Fragment key={group.id}>
                        <tr
                          className="group-row"
                          onClick={() => setExpandedRows(p =>
                            p.includes(group.id) ? p.filter(r => r !== group.id) : [...p, group.id]
                          )}
                        >
                          <td style={{ fontWeight: 600, paddingLeft: 20 }}>
                            <span style={{ color: 'var(--text-3)', marginRight: 8 }}>
                              {isExpanded ? '▾' : '▸'}
                            </span>
                            {group.label}
                            {group.totalJobs > 1 && (
                              <span style={{ fontSize: '0.78em', color: 'var(--text-3)', marginLeft: 6 }}>
                                ({group.totalJobs} sets)
                              </span>
                            )}
                          </td>
                          <td>
                            <span className={`status-dot ${group.status}`}>
                              {group.status === 'done' ? 'Done' : group.status === 'running' ? 'Running' : group.status === 'failed' ? 'Failed' : 'Pending'}
                            </span>
                          </td>
                          <td><Cell status={group.status} val={group.metrics.result_match}    metricKey="accuracy" /></td>
                          <td><Cell status={group.status} val={group.metrics.semantic_judge}  metricKey="accuracy" /></td>
                          <td><Cell status={group.status} val={group.metrics.retrieval_recall} metricKey="recall" /></td>
                          <td><Cell status={group.status} val={group.metrics.avg_attempts}    metricKey="attempts" /></td>
                          <td><Cell status={group.status} val={group.metrics.avg_tool_calls}  metricKey="tool" /></td>
                          <td><Cell status={group.status} val={group.metrics.avg_latency}     metricKey="latency" /></td>
                          <td><Cell status={group.status} val={group.metrics.avg_cost}        metricKey="cost" /></td>
                        </tr>

                        {isExpanded && group.children.map(job => (
                          <tr key={`${job.strategy}-${job.difficulty}`} className="child-row">
                            <td style={{ paddingLeft: 48, color: 'var(--text-3)', fontSize: '0.85em' }}>
                              ↳ {job.difficulty}
                            </td>
                            <td>
                              <span className={`status-dot ${job.status === 'completed' ? 'done' : job.status}`}>
                                {job.status === 'completed' ? 'Done' : job.status === 'running' ? 'Running' : job.status === 'failed' ? 'Failed' : 'Pending'}
                              </span>
                            </td>
                            <td><Cell status={job.status} val={getVal(job.results, 'result_match.accuracy')}    metricKey="accuracy" /></td>
                            <td><Cell status={job.status} val={getVal(job.results, 'semantic_judge.accuracy')}  metricKey="accuracy" /></td>
                            <td><Cell status={job.status} val={getVal(job.results, 'retrieval_recall.mean')}    metricKey="recall" /></td>
                            <td><Cell status={job.status} val={getVal(job.results, 'avg_attempts.mean')}        metricKey="attempts" /></td>
                            <td><Cell status={job.status} val={getVal(job.results, 'avg_tool_calls.mean')}      metricKey="tool" /></td>
                            <td><Cell status={job.status} val={getVal(job.results, 'avg_latency.mean')}         metricKey="latency" /></td>
                            <td><Cell status={job.status} val={getVal(job.results, 'avg_cost.mean')}            metricKey="cost" /></td>
                          </tr>
                        ))}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <h3>No benchmark data yet</h3>
              <p>Select strategies and difficulties on the left, then click Run Evals.</p>
            </div>
          )}
        </div>

      )}
    </div>
  );
}
