import { useEffect, useRef, useState } from 'react'
import { analyzeFile, type AnalysisReport } from './api'
import { ChecksTable } from './components/ChecksTable'
import { EvidenceImages } from './components/EvidenceImages'
import { FindingsList } from './components/FindingsList'
import { ResultSummary } from './components/ResultSummary'
import { UploadDropzone } from './components/UploadDropzone'

type State =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'done'; report: AnalysisReport }

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [state, setState] = useState<State>({ kind: 'idle' })
  const request = useRef<AbortController | null>(null)

  // Release the preview URL when the file changes.
  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview)
  }, [preview])

  async function handleFile(next: File) {
    request.current?.abort()
    const controller = new AbortController()
    request.current = controller

    setFile(next)
    setPreview(URL.createObjectURL(next))
    setState({ kind: 'loading' })
    try {
      const report = await analyzeFile(next, controller.signal)
      setState({ kind: 'done', report })
    } catch (error) {
      if (controller.signal.aborted) return
      setState({ kind: 'error', message: error instanceof Error ? error.message : String(error) })
    }
  }

  const isVideo = file?.type.startsWith('video/')
  const warnings =
    state.kind === 'done' ? [...state.report.metadata.warnings, ...state.report.warnings] : []

  return (
    <main className="container">
      <header className="hud-header">
        <span className="hud-tag">
          <span className="status-dot" />
          MEDIA INTEGRITY SCANNER
        </span>
        <h1>Forensics Analyzer</h1>
        <p className="muted">
          Upload a photo or video to check its metadata and pixels for signs of editing.
        </p>
      </header>

      <UploadDropzone onFile={handleFile} disabled={state.kind === 'loading'} />

      {file && preview && (
        <section className="card preview">
          {isVideo ? (
            <video src={preview} controls muted />
          ) : (
            <img src={preview} alt={`Preview of ${file.name}`} />
          )}
        </section>
      )}

      {state.kind === 'loading' && (
        <div className="status-line" aria-live="polite">
          <span>&gt; SCANNING {file?.name}…</span>
          <div className="scan-bar" />
        </div>
      )}

      {state.kind === 'error' && (
        <section className="card error" role="alert">
          <h3>Could not analyze this file</h3>
          <p>{state.message}</p>
        </section>
      )}

      {state.kind === 'done' && (
        <>
          <ResultSummary report={state.report} />
          <FindingsList findings={state.report.metadata.findings} />
          {state.report.image && <EvidenceImages image={state.report.image} />}
          {state.report.video_detectors && (
            <section className="card">
              <h3>Frame analysis</h3>
              <p className="muted">
                Frame-level video detectors are not built yet. These results come from the metadata
                checks only.
              </p>
            </section>
          )}
          <ChecksTable results={state.report.metadata.results} />
          {warnings.length > 0 && (
            <section className="card">
              <h3>Warnings</h3>
              <ul>
                {warnings.map((w, i) => (
                  <li key={i} className="muted">
                    {w}
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </main>
  )
}
