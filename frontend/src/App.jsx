import React, { useState, useRef, useEffect } from 'react';
import './index.css';

const STRATEGIES = [
  { id: 'zero_shot', label: 'Zero Shot (Baseline)' },
  { id: 'few_shot_static', label: 'Few Shot (Static)' },
  { id: 'few_shot_dynamic', label: 'Few Shot (Dynamic)' },
  { id: 'chain_of_thought', label: 'Chain of Thought' },
  { id: 'rag', label: 'Vector RAG' },
  { id: 'dspy', label: 'DSPy Optimized' },
  { id: 'routed', label: 'Smart Router' }
];

const DIFFICULTIES = ['easy', 'medium', 'hard'];

function App() {
  const [activeTab, setActiveTab] = useState('eval'); // 'chat' or 'eval'

  // Chat State
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      text: "Hello! I am your Enterprise Text-to-SQL assistant. I am connected to your DuckDB Sandbox.\n\nAsk me questions like:\n- What is the total revenue per category?\n- Which customers are in the USA?"
    }
  ]);
  const [input, setInput] = useState('');
  const [chatStrategy, setChatStrategy] = useState('dspy');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  // Eval State
  const [selectedDifficulties, setSelectedDifficulties] = useState(['hard']);
  const [selectedStrategies, setSelectedStrategies] = useState(['rag', 'dspy']);
  
  // Dashboard Table State
  const [expandedRows, setExpandedRows] = useState([]);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'desc' });
  const activePolls = useRef({});

  // -------------------------------------------------------------
  // Persistent Cache Logic (Best Practice for browser-based state)
  // -------------------------------------------------------------
  const [jobs, setJobs] = useState(() => {
    // Initialize lazily from LocalStorage
    const saved = localStorage.getItem('sqlEvalJobsCache');
    if (saved) return JSON.parse(saved);
    return [];
  });

  useEffect(() => {
    // Flush to LocalStorage whenever jobs object changes
    const completedOrFailedJobs = jobs.filter(j => j.status === 'completed' || j.status === 'failed');
    localStorage.setItem('sqlEvalJobsCache', JSON.stringify(completedOrFailedJobs));
  }, [jobs]);

  const clearCache = () => {
    setJobs([]);
    localStorage.removeItem('sqlEvalJobsCache');
  };
  // -------------------------------------------------------------

  useEffect(() => {
    if (activeTab === 'chat' && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, activeTab]);

  const toggleStrategy = (id) => {
    setSelectedStrategies(prev => 
      prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
    );
  };

  const toggleDifficulty = (diff) => {
    setSelectedDifficulties(prev => 
      prev.includes(diff) ? prev.filter(d => d !== diff) : [...prev, diff]
    );
  };

  const handleSubmitChat = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuestion = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userQuestion }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userQuestion, strategy: chatStrategy })
      });
      
      const data = await response.json();
      
      setMessages(prev => [...prev, {
        role: 'ai',
        sql: data.sql,
        reasoning: data.reasoning,
        data: data.data,
        metrics: {
          latency: data.latency,
          cost: data.cost,
          tokens: data.completion_tokens + data.prompt_tokens,
          router: data.routed_difficulty ? `Router selected: ${data.router_method}` : null
        }
      }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', text: `Error connecting to API: ${error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const startBatchEval = async () => {
    if (selectedStrategies.length === 0 || selectedDifficulties.length === 0) return;
    
    setExpandedRows(selectedStrategies); // Expand the UI automatically
    
    // Matrix of all combinations requested
    const requestedCombinations = [];
    selectedStrategies.forEach(strat => {
       selectedDifficulties.forEach(diff => {
          requestedCombinations.push({ strat, diff });
       });
    });

    // 1. Differentiate between already Cached (completed) vs Missing/Failed
    const jobsToExecute = [];
    const jobsToKeep = [...jobs];

    requestedCombinations.forEach(combo => {
       const existingIndex = jobsToKeep.findIndex(j => j.strategy === combo.strat && j.difficulty === combo.diff);
       
       if (existingIndex !== -1 && jobsToKeep[existingIndex].status === 'completed') {
          // Already have it successfully cached! Do nothing.
       } else {
          // It's missing or failed. Add to execute queue.
          jobsToExecute.push(combo);
          
          // Re-initialize state slot (remove old failed state if it existed)
          if (existingIndex !== -1) {
             jobsToKeep[existingIndex] = { ...jobsToKeep[existingIndex], status: 'pending', jobId: null, results: null };
          } else {
             jobsToKeep.push({ jobId: null, strategy: combo.strat, difficulty: combo.diff, status: 'pending', results: null });
          }
       }
    });

    setJobs(jobsToKeep);

    if (jobsToExecute.length === 0) {
       // Everything they clicked is already downloaded! Instant UI.
       return;
    }
    
    // 2. Dispatch the Missing Jobs to backend
    const dispatchedJobs = await Promise.all(jobsToExecute.map(async (combo) => {
      try {
        const res = await fetch('http://127.0.0.1:8000/evals/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ strategy: combo.strat, difficulty: combo.diff })
        });
        const data = await res.json();
        return { strategy: combo.strat, difficulty: combo.diff, jobId: data.job_id, status: 'running' };
      } catch (e) {
        return { strategy: combo.strat, difficulty: combo.diff, jobId: null, status: 'failed' };
      }
    }));
    
    // 3. Setup polling listeners
    setJobs(currentJobs => 
      currentJobs.map(job => {
        const dispatch = dispatchedJobs.find(d => d.strategy === job.strategy && d.difficulty === job.difficulty);
        if (dispatch && dispatch.jobId) {
           pollJob(dispatch.jobId, job.strategy, job.difficulty);
           return { ...job, jobId: dispatch.jobId, status: 'running' };
        }
        // If it was in the execute queue but dispatch failed, mark failed
        const wasInExecuteQueue = jobsToExecute.find(d => d.strat === job.strategy && d.diff === job.difficulty);
        if (wasInExecuteQueue && !dispatch) return { ...job, status: 'failed' };
        
        return job; // leave existing cached jobs untouched
      })
    );
  };

  const pollJob = (jobId) => {
    const id = setInterval(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/evals/${jobId}`);
        const data = await res.json();
        
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(id);
          delete activePolls.current[jobId];
          
          setJobs(currentJobs => currentJobs.map(job => {
            if (job.jobId === jobId) {
               return { 
                 ...job, 
                 status: data.status, 
                 results: data.results?.scores || null 
               };
            }
            return job;
          }));
        }
      } catch (e) {
        clearInterval(id);
        delete activePolls.current[jobId];
        
        setJobs(currentJobs => currentJobs.map(job => 
          job.jobId === jobId ? { ...job, status: 'failed' } : job
        ));
      }
    }, 3000);
    
    activePolls.current[jobId] = id;
  };

  const renderTable = (rows) => {
    if (!rows || rows.length === 0) return <p style={{color: '#94a3b8', fontStyle: 'italic'}}>No rows returned.</p>;
    if (rows.length === 1 && rows[0].error) {
       return <p style={{color: '#ef4444'}}>DuckDB Error: {rows[0].error}</p>;
    }
    
    const headers = Object.keys(rows[0]);
    return (
      <div style={{overflowX: 'auto'}}>
        <table className="data-table">
          <thead>
            <tr>{headers.map(h => <th key={h}>{h}</th>)}</tr>
          </thead>
          <tbody>
             {rows.map((row, i) => (
              <tr key={i}>
                 {headers.map(h => <td key={h}>{String(row[h])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const isAnyJobRunning = jobs.some(j => j.status === 'running' || j.status === 'pending');

  const handleSort = (key) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const sortIcon = (key) => {
    if (sortConfig.key !== key) return null;
    return sortConfig.direction === 'asc' ? ' ▴' : ' ▾';
  };

  const groupedJobs = STRATEGIES.filter(s => selectedStrategies.includes(s.id)).map(st => {
    const gJobs = jobs.filter(j => j.strategy === st.id);
    
    const completedJobs = gJobs.filter(j => j.status === 'completed' && j.results);
    const isRunning = gJobs.some(j => ['pending', 'running'].includes(j.status));
    const isFailed = gJobs.some(j => j.status === 'failed');
    const status = isRunning ? 'running' : (isFailed ? 'failed' : (gJobs.length > 0 ? 'completed' : 'empty'));
    
    const avgScore = (path) => {
       if (completedJobs.length === 0) return null;
       let sum = 0;
       let count = 0;
       completedJobs.forEach(job => {
          const val = path.split('.').reduce((o, i) => o ? o[i] : null, job.results);
          if (typeof val === 'number') { sum += val; count++; }
       });
       return count === 0 ? null : sum / count;
    };

    return {
      strategy: st.id,
      label: st.label,
      summaryStatus: status,
      totalJobs: gJobs.length,
      children: gJobs,
      metrics: {
        result_match: avgScore('result_match.accuracy'),
        semantic_judge: avgScore('semantic_judge.accuracy'),
        retrieval_recall: avgScore('retrieval_recall.mean'),
        avg_latency: avgScore('avg_latency.mean'),
        avg_cost: avgScore('avg_cost.mean')
      }
    };
  }).filter(g => g.totalJobs > 0); // only show groups that have actual jobs cached or running

  const sortedGroups = [...groupedJobs].sort((a, b) => {
    if (!sortConfig.key) return 0;
    if (sortConfig.key === 'strategy') {
       const res = a.label.localeCompare(b.label);
       return sortConfig.direction === 'asc' ? res : -res;
    }
    if (sortConfig.key === 'status') {
       const res = a.summaryStatus.localeCompare(b.summaryStatus);
       return sortConfig.direction === 'asc' ? res : -res;
    }
    
    const valA = a.metrics[sortConfig.key];
    const valB = b.metrics[sortConfig.key];
    
    if (valA === null && valB !== null) return 1;
    if (valA !== null && valB === null) return -1;
    if (valA === null && valB === null) return 0;
    
    return sortConfig.direction === 'asc' ? valA - valB : valB - valA;
  });

  const formatVal = (val, key) => {
     if (val === null || val === undefined) return '-';
     if (key.includes('accuracy') || key.includes('recall')) return (val * 100).toFixed(1) + '%';
     if (key.includes('latency')) return val.toFixed(2) + 's';
     if (key.includes('cost')) return '$' + val.toFixed(4);
     return val;
  };

  const RawValueRenderer = ({ status, val, keyPath }) => {
    if (status === 'running' || status === 'pending') return <span style={{color: '#94a3b8'}}>...</span>;
    if (status === 'failed') return <span style={{color: '#ef4444'}}>Error</span>;
    return <span>{formatVal(val, keyPath)}</span>;
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div>
          <h2>Evaluation Lab</h2>
          <p style={{fontSize: '0.85em', color: 'var(--text-muted)'}}>Enterprise Text-to-SQL</p>
        </div>

        <div className="tabs" style={{marginTop: '20px'}}>
          <div className={`tab ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
            Chat
          </div>
          <div className={`tab ${activeTab === 'eval' ? 'active' : ''}`} onClick={() => setActiveTab('eval')}>
            Leaderboard
          </div>
        </div>
        
        {activeTab === 'chat' && (
          <div className="control-group">
            <label>Prompting Strategy</label>
            <select value={chatStrategy} onChange={e => setChatStrategy(e.target.value)}>
              {STRATEGIES.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
          </div>
        )}

        {activeTab === 'eval' && (
          <>
            <div className="control-group">
              <label>Dataset Difficulty</label>
              <div style={{display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '12px'}}>
                {DIFFICULTIES.map(d => (
                  <label key={d} className="checkbox-label" style={{display: 'flex', alignItems: 'center', gap: '8px', textTransform: 'none', color: 'var(--text-main)', cursor: 'pointer', fontSize: '0.9em', margin: 0}}>
                    <input 
                      type="checkbox" 
                      style={{accentColor: 'var(--accent)', transform: 'scale(1.2)'}}
                      checked={selectedDifficulties.includes(d)}
                      onChange={() => toggleDifficulty(d)}
                      disabled={isAnyJobRunning}
                    />
                    {d.toUpperCase()}
                  </label>
                ))}
              </div>
            </div>
            
            <div className="control-group">
              <label>Compare Strategies</label>
              <div style={{display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '12px'}}>
                {STRATEGIES.map(s => (
                  <label key={s.id} className="checkbox-label" style={{display: 'flex', alignItems: 'center', gap: '8px', textTransform: 'none', color: 'var(--text-main)', cursor: 'pointer', fontSize: '0.9em', margin: 0}}>
                    <input 
                      type="checkbox" 
                      style={{accentColor: 'var(--accent)', transform: 'scale(1.2)'}}
                      checked={selectedStrategies.includes(s.id)}
                      onChange={() => toggleStrategy(s.id)}
                      disabled={isAnyJobRunning}
                    />
                    {s.label}
                  </label>
                ))}
              </div>
            </div>
            
            <div style={{display: 'flex', gap: '10px', marginTop: '10px'}}>
               <button 
                 className="btn-primary" 
                 onClick={startBatchEval} 
                 disabled={isAnyJobRunning || selectedStrategies.length === 0 || selectedDifficulties.length === 0} 
                 style={{flex: 1}}>
                 {isAnyJobRunning ? 'Running...' : 'Run Selected'}
               </button>
               <button 
                 onClick={clearCache} 
                 style={{background: 'rgba(239, 68, 68, 0.2)', color: '#ef4444', border: 'none', borderRadius: '16px', padding: '0 16px', cursor: 'pointer'}}
                 title="Clear Cache">
                 🗑️
               </button>
            </div>
          </>
        )}

        <div style={{marginTop: 'auto', fontSize: '0.8em', color: 'var(--text-muted)'}}>
          Connected to local DuckDB and DSPy backend.
        </div>
      </div>
      
      {activeTab === 'chat' ? (
        <div className="chat-container">
          <div className="messages" ref={scrollRef}>
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role === 'user' ? 'user-msg' : 'ai-msg'}`}>
                {msg.role === 'user' ? (
                  <div>{msg.text}</div>
                ) : (
                  <div className="ai-content">
                    {msg.text && <p style={{whiteSpace: 'pre-wrap'}}>{msg.text}</p>}
                    
                    {msg.reasoning && (
                      <div className="reasoning-box">
                        {msg.reasoning}
                      </div>
                    )}
                    
                    {msg.sql && (
                      <div className="sql-box">
                        {msg.sql}
                      </div>
                    )}
                    
                    {msg.data !== undefined && renderTable(msg.data)}
                    
                    {msg.metrics && (
                      <div className="metrics" style={{flexWrap: 'wrap'}}>
                        <span className="metric-badge">{msg.metrics.latency.toFixed(2)}s Latency</span>
                        <span className="metric-badge">${msg.metrics.cost.toFixed(5)}</span>
                        {msg.metrics.tokens > 0 && <span className="metric-badge">{msg.metrics.tokens} Tokens</span>}
                        {msg.metrics.router && <span className="metric-badge" style={{borderColor: '#818cf8', color: '#818cf8'}}>{msg.metrics.router}</span>}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="message ai-msg">
                  <span style={{color: '#94a3b8', fontStyle: 'italic'}}>Generating SQL...</span>
              </div>
            )}
          </div>
          
          <div className="input-area">
            <form className="input-box" onSubmit={handleSubmitChat}>
              <input 
                type="text" 
                placeholder="Ask a question about the data..." 
                value={input}
                onChange={e => setInput(e.target.value)}
                disabled={isLoading}
              />
              <button type="submit" disabled={isLoading || !input.trim()}>
                Send
              </button>
            </form>
          </div>
        </div>
      ) : (
        <div className="eval-dashboard">
          <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px'}}>
             <h1 style={{margin: 0}}>Evaluation Leaderboard</h1>
             {isAnyJobRunning && <div className="spinner" style={{margin: 0, width: '24px', height: '24px', borderWidth: '3px'}}></div>}
          </div>
          <p style={{color: 'var(--text-muted)', marginBottom: '10px'}}>
            Select multiple strategies and difficulties on the left. Click column headers to sort the matrix. 
            Cached results load instantly. Click the trash can to force a re-run.
          </p>

          {jobs.length > 0 ? (
            <div style={{overflowX: 'auto', marginTop: '20px'}}>
              <table className="data-table" style={{width: '100%', minWidth: '900px', fontSize: '1em'}}>
                <thead>
                  <tr style={{cursor: 'pointer'}}>
                    <th onClick={() => handleSort('strategy')}>Strategy {sortIcon('strategy')}</th>
                    <th onClick={() => handleSort('status')}>Status {sortIcon('status')}</th>
                    <th onClick={() => handleSort('result_match')}>Result Match {sortIcon('result_match')}</th>
                    <th onClick={() => handleSort('semantic_judge')}>Semantic Score {sortIcon('semantic_judge')}</th>
                    <th onClick={() => handleSort('retrieval_recall')}>RAG Recall {sortIcon('retrieval_recall')}</th>
                    <th onClick={() => handleSort('avg_latency')}>Latency {sortIcon('avg_latency')}</th>
                    <th onClick={() => handleSort('avg_cost')}>Cost {sortIcon('avg_cost')}</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedGroups.map((group) => {
                     const isExpanded = expandedRows.includes(group.strategy);
                     
                     return (
                      <React.Fragment key={group.strategy}>
                        {/* Parent Aggregate Row */}
                        <tr 
                          onClick={() => setExpandedRows(prev => prev.includes(group.strategy) ? prev.filter(r => r !== group.strategy) : [...prev, group.strategy])}
                          style={{
                            background: group.summaryStatus === 'running' ? 'rgba(59, 130, 246, 0.05)' : 'rgba(255, 255, 255, 0.03)',
                            cursor: 'pointer',
                            borderBottom: isExpanded ? 'none' : ''
                          }}>
                          <td style={{fontWeight: '600', paddingLeft: '20px'}}>
                            <span style={{display:'inline-block', width:'20px', color: 'var(--text-muted)'}}>{isExpanded ? '▾' : '▸'}</span> 
                            {group.label} {group.totalJobs > 1 && <span style={{fontSize: '0.8em', color: '#94a3b8', marginLeft: '6px'}}>({group.totalJobs} sets)</span>}
                          </td>
                          <td>
                             {group.summaryStatus === 'running' ? <span style={{color: '#60a5fa'}}>Running</span> : 
                              group.summaryStatus === 'failed' ? <span style={{color: '#ef4444'}}>Failed</span> : 
                              <span style={{color: '#4ade80'}}>Done</span>}
                          </td>
                          <td style={{fontWeight: '500'}}><RawValueRenderer status={group.summaryStatus} val={group.metrics.result_match} keyPath="result_match.accuracy" /></td>
                          <td style={{fontWeight: '500'}}><RawValueRenderer status={group.summaryStatus} val={group.metrics.semantic_judge} keyPath="semantic_judge.accuracy" /></td>
                          <td style={{fontWeight: '500'}}><RawValueRenderer status={group.summaryStatus} val={group.metrics.retrieval_recall} keyPath="retrieval_recall.mean" /></td>
                          <td style={{fontWeight: '500'}}><RawValueRenderer status={group.summaryStatus} val={group.metrics.avg_latency} keyPath="avg_latency.mean" /></td>
                          <td style={{fontWeight: '500'}}><RawValueRenderer status={group.summaryStatus} val={group.metrics.avg_cost} keyPath="avg_cost.mean" /></td>
                        </tr>

                        {/* Child Detail Rows */}
                        {isExpanded && group.children.map(job => {
                           const valPath = (path) => job.results ? path.split('.').reduce((o, i) => o ? o[i] : null, job.results) : null;
                           
                           return (
                             <tr key={`${job.strategy}-${job.difficulty}`} style={{background: 'rgba(0,0,0,0.1)'}}>
                                <td style={{paddingLeft: '50px', color: '#94a3b8', fontSize: '0.9em', textTransform: 'capitalize'}}>↳ Dataset: {job.difficulty}</td>
                                <td style={{fontSize: '0.9em'}}>
                                   {job.status === 'running' || job.status === 'pending' ? <span style={{color: '#60a5fa'}}>Queued</span> : 
                                    job.status === 'failed' ? <span style={{color: '#ef4444'}}>Failed</span> : 
                                    <span style={{color: '#4ade80'}}>Done</span>}
                                </td>
                                <td><RawValueRenderer status={job.status} val={valPath('result_match.accuracy')} keyPath="result_match.accuracy" /></td>
                                <td><RawValueRenderer status={job.status} val={valPath('semantic_judge.accuracy')} keyPath="semantic_judge.accuracy" /></td>
                                <td><RawValueRenderer status={job.status} val={valPath('retrieval_recall.mean')} keyPath="retrieval_recall.mean" /></td>
                                <td><RawValueRenderer status={job.status} val={valPath('avg_latency.mean')} keyPath="avg_latency.mean" /></td>
                                <td><RawValueRenderer status={job.status} val={valPath('avg_cost.mean')} keyPath="avg_cost.mean" /></td>
                             </tr>
                           );
                        })}
                      </React.Fragment>
                     );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{textAlign: 'center', padding: '60px', border: '1px dashed var(--border-color)', borderRadius: '12px', marginTop: '40px'}}>
              <h3 style={{color: 'var(--text-muted)'}}>No Benchmark Data</h3>
              <p style={{color: '#475569', fontSize: '0.9em', marginTop: '8px'}}>Select combinations on the left and run a batch.</p>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

export default App;
