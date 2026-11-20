import { useEffect, useRef, useState } from 'react'
import { analyzeFile, summarize, type AnalysisReport, type Summary } from './api'
import { AiSummary } from './components/AiSummary'
import { CategorySection } from './components/CategorySection'
import { UploadDropzone } from './components/UploadDropzone'
import { VerdictBanner } from './components/VerdictBanner'
import { CATEGORY_ORDER } from './format'

type AnalysisState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'done'; report: AnalysisReport }

type SummaryState =
  | { state: 'loading' }
  | { state: 'error'; message: string }
  | { state: 'done'; summary: Summary }

const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error))

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisState>({ kind: 'idle' })
  const [summary, setSummary] = useState<SummaryState>({ state: 'loading' })
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
    setAnalysis({ kind: 'loading' })
    setSummary({ state: 'loading' })

    let report: AnalysisReport
    try {
      report = await analyzeFile(next, controller.signal)
      setAnalysis({ kind: 'done', report })
    } catch (error) {
      if (!controller.signal.aborted) setAnalysis({ kind: 'error', message: errorMessage(error) })
      return
    }

    // Show the results straight away, then fill in the summary when it arrives.
    try {
      setSummary({ state: 'done', summary: await summarize(report, controller.signal) })
    } catch (error) {
      if (!controller.signal.aborted) setSummary({ state: 'error', message: errorMessage(error) })
    }
  }

  const isVideo = file?.type.startsWith('video/')

  return (
    <main className="container">
      <header className="hud-header">
        <span className="hud-tag">
          <span className="status-dot" />
          MEDIA INTEGRITY SCANNER
        </span>
        <h1>Forensics Analyzer</h1>
        <p className="muted">
          Upload a photo or video to check its metadata, pixels, compression and frames for signs
          of editing.
        </p>
      </header>

      <div className="upload-row">
        <UploadDropzone onFile={handleFile} disabled={analysis.kind === 'loading'} />
        {file && preview && (
          <section className="card preview">
            {isVideo ? (
              <video src={preview} controls muted />
            ) : (
              <img src={preview} alt={`Preview of ${file.name}`} />
            )}
          </section>
        )}
      </div>

      {analysis.kind === 'loading' && (
        <div className="status-line" aria-live="polite">
          <span>&gt; SCANNING {file?.name}…</span>
          <div className="scan-bar" />
        </div>
      )}

      {analysis.kind === 'error' && (
        <section className="card error" role="alert">
          <h3>Could not analyze this file</h3>
          <p>{analysis.message}</p>
        </section>
      )}

      {analysis.kind === 'done' && (
        <>
          <VerdictBanner report={analysis.report} />
          <AiSummary {...summary} />
          {CATEGORY_ORDER.map((key) => (
            <CategorySection key={key} id={key} category={analysis.report.categories[key]} />
          ))}
          {analysis.report.warnings.length > 0 && (
            <section className="card">
              <h3>Warnings</h3>
              <ul>
                {analysis.report.warnings.map((w, i) => (
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
