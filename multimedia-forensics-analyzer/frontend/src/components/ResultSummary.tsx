import type { AnalysisReport } from '../api'

function level(score: number): 'low' | 'medium' | 'high' {
  if (score >= 0.6) return 'high'
  if (score >= 0.3) return 'medium'
  return 'low'
}

export function ResultSummary({ report }: { report: AnalysisReport }) {
  const { metadata } = report
  const score = metadata.score
  const lvl = level(score)

  return (
    <section className={`card summary summary-${lvl}`}>
      <div className="summary-head">
        <div>
          <h2>{report.file_name}</h2>
          <p className="muted">
            {report.media_type} · {metadata.checks_run} of {metadata.checks_total} metadata checks
            could run
          </p>
        </div>
        <div className="score" aria-label={`Suspicion score ${score.toFixed(2)} out of 1`}>
          <span className="score-value">{score.toFixed(2)}</span>
          <span className="score-label">Suspicion score</span>
        </div>
      </div>

      <div className="meter" role="presentation">
        <div className="meter-fill" style={{ width: `${Math.round(score * 100)}%` }} />
      </div>

      <p className="conclusion">{metadata.conclusion}</p>
      <p className="muted small">
        The score measures evidence of editing found in the file. A low score does not prove the
        file is original.
      </p>
    </section>
  )
}
