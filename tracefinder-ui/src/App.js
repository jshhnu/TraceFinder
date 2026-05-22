import React, { useState } from 'react';

// Consolidated 7-class visual dashboard tracker
const SCANNER_MAPPING = {
  0: "Canon LiDE 120 (Units 1 & 2)",
  1: "Canon LiDE 220",
  2: "Canon 9000F (Units 1 & 2)",
  3: "Epson Perfection V39 (Units 1 & 2)",
  4: "Epson Perfection V370 (Units 1 & 2)",
  5: "Epson Perfection V550",
  6: "HP ScanJet Pro Series"
};

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResults(null);
    setError(null);
  };

  const executeForensicAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (data.success) {
        setResults(data.predictions);
      } else {
        setError(data.error || "Analysis failed.");
      }
    } catch (err) {
      setError("Cannot establish link to GPU backend API gateway. Ensure app.py is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.appContainer}>
      {/* Top Banner Navigation */}
      <header style={styles.navbar}>
        <div style={styles.logoGroup}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          <h1 style={styles.navTitle}>TraceFinder <span style={styles.versionBadge}>v2.0 Deep DL</span></h1>
        </div>
        <div style={styles.hardwareIndicator}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>
            <rect x="9" y="9" width="6" height="6"/>
            <line x1="9" y1="1" x2="9" y2="4"/>
            <line x1="15" y1="1" x2="15" y2="4"/>
            <line x1="9" y1="20" x2="9" y2="23"/>
            <line x1="15" y1="20" x2="15" y2="23"/>
            <line x1="20" y1="9" x2="23" y2="9"/>
            <line x1="20" y1="15" x2="23" y2="15"/>
            <line x1="1" y1="9" x2="4" y2="9"/>
            <line x1="1" y1="15" x2="4" y2="15"/>
          </svg>
          <span style={styles.hardwareText}>RTX 3050 Accelerator Linked</span>
        </div>
      </header>

      {/* Main Grid Workspace */}
      <main style={styles.dashboardGrid}>

        {/* Left Column: Forensic Dropzone Panel */}
        <section style={styles.panelCard}>
          <h2 style={styles.sectionHeader}>Forensic Input Processing</h2>
          <p style={styles.subtext}>Upload a standardized high-frequency patch array file (.npy) extracted from device memory to parse hardware sensor noise signatures.</p>

          <div style={styles.uploadBox}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginBottom: '16px' }}>
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <input type="file" accept=".npy" onChange={handleFileChange} style={styles.fileInput} id="patch-upload" />
            <label htmlFor="patch-upload" style={styles.uploadLabel}>
              {file ? file.name : "Select or Drop Test Patch Array"}
            </label>
          </div>

          <button
            onClick={executeForensicAnalysis}
            disabled={!file || loading}
            style={{...styles.actionButton, opacity: (!file || loading) ? 0.6 : 1}}
          >
            {loading ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span className="spin-element" style={styles.spinnerCore}></span>
                Parsing High-Frequency Residuals...
              </span>
            ) : "Run Extraction & Verification"}
          </button>

          {error && <div style={styles.errorBanner}>{error}</div>}
        </section>

        {/* Right Column: Dynamic Results Dashboard */}
        <section style={styles.panelCard}>
          <h2 style={styles.sectionHeader}>Classifier Metrics Output</h2>

          {!results && !loading && (
            <div style={styles.emptyState}>
              <p>Awaiting data telemetry. Load a sample signature input to evaluate the physical micro-anomalies.</p>
            </div>
          )}

          {loading && (
            <div style={styles.emptyState}>
              <span className="spin-element" style={{...styles.spinnerCore, width: '32px', height: '32px', borderLeftColor: '#3b82f6' }}></span>
              <p style={{ marginTop: '16px' }}>Executing native 5x5 Wiener deconvolution on device tensor channels...</p>
            </div>
          )}

          {results && (
            <div style={styles.resultsWrapper}>
              <div style={styles.matchBanner}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                  <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                <span style={styles.matchText}>Signature Match Identified</span>
              </div>

              <div style={styles.metricsContainer}>
                {results.map((item, idx) => (
                  <div key={idx} style={styles.metricRow}>
                    <div style={styles.metricLabels}>
                      <span style={{...styles.scannerName, fontWeight: idx === 0 ? '600' : '400', color: idx === 0 ? '#f8fafc' : '#94a3b8'}}>
                        {item.scanner}
                      </span>
                      <span style={{...styles.confidenceScore, color: idx === 0 ? '#34d399' : '#64748b'}}>
                        {item.confidence.toFixed(2)}%
                      </span>
                    </div>
                    <div style={styles.barTrack}>
                      <div style={{
                        ...styles.barFill,
                        width: `${item.confidence}%`,
                        backgroundColor: idx === 0 ? '#2563eb' : '#334155'
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Embedded CSS Loader Animations */}
      <style dangerouslySetInnerHTML={{__html: `
        .spin-element {
          display: inline-block;
          border: 3px solid rgba(255,255,255,0.1);
          border-radius: 50%;
          border-left-color: #fff;
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}} />
    </div>
  );
}

// Layout Styling Parameters
const styles = {
  appContainer: { backgroundColor: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif' },
  navbar: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 40px', backgroundColor: '#1e293b', borderBottom: '1px solid #334155' },
  logoGroup: { display: 'flex', alignItems: 'center', gap: '12px' },
  navTitle: { fontSize: '20px', fontWeight: '700', color: '#f8fafc', margin: 0 },
  versionBadge: { fontSize: '11px', color: '#3b82f6', backgroundColor: '#1e3a8a', padding: '2px 8px', borderRadius: '12px', marginLeft: '6px' },
  hardwareIndicator: { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#064e3b', padding: '6px 12px', borderRadius: '6px', border: '1px solid #065f46' },
  hardwareText: { fontSize: '13px', color: '#34d399', fontWeight: '500' },
  dashboardGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', padding: '40px', maxWidth: '1400px', margin: '0 auto' },
  panelCard: { backgroundColor: '#1e293b', borderRadius: '12px', padding: '32px', border: '1px solid #334155', display: 'flex', flexDirection: 'column' },
  sectionHeader: { fontSize: '18px', fontWeight: '600', color: '#f8fafc', marginTop: 0, marginBottom: '8px' },
  subtext: { fontSize: '13px', color: '#94a3b8', lineHeight: '1.5', marginBottom: '24px' },
  uploadBox: { border: '2px dashed #475569', borderRadius: '8px', padding: '40px 20px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', backgroundColor: '#0f172a', marginBottom: '24px', position: 'relative' },
  fileInput: { position: 'absolute', opacity: 0, width: '100%', height: '100%', cursor: 'pointer' },
  uploadLabel: { fontSize: '14px', color: '#3b82f6', fontWeight: '500', cursor: 'pointer', textAlign: 'center' },
  actionButton: { display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#2563eb', color: '#ffffff', border: 'none', padding: '12px 24px', borderRadius: '6px', fontWeight: '600', fontSize: '14px', cursor: 'pointer' },
  spinnerCore: { width: '16px', height: '16px' },
  errorBanner: { marginTop: '16px', padding: '12px', backgroundColor: '#7f1d1d', border: '1px solid #991b1b', borderRadius: '6px', color: '#fca5a5', fontSize: '13px' },
  emptyState: { flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: '#64748b', fontSize: '14px', padding: '40px' },
  resultsWrapper: { display: 'flex', flexDirection: 'column', gap: '20px' },
  matchBanner: { display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#064e3b', padding: '10px 16px', borderRadius: '6px', border: '1px solid #065f46' },
  matchText: { fontSize: '14px', color: '#34d399', fontWeight: '600' },
  metricsContainer: { display: 'flex', flexDirection: 'column', gap: '16px' },
  metricRow: { display: 'flex', flexDirection: 'column', gap: '6px' },
  metricLabels: { display: 'flex', justifyContent: 'space-between', fontSize: '14px' },
  scannerName: { fontSize: '14px' },
  confidenceScore: { fontWeight: '600' },
  barTrack: { height: '8px', backgroundColor: '#0f172a', borderRadius: '4px', overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: '4px', transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)' }
};

export default App;