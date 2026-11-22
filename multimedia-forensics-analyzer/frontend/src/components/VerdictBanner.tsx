import type { AnalysisReport } from '../api'
import { CATEGORY_ORDER, CATEGORY_SHORT, VERDICT_LABELS } from '../format'

function chipText(status: string, flagged: number): string {
  if (status === 'flagged') return `${flagged} flag${flagged === 1 ? '' : 's'}`
  if (status === 'clear') return 'clear'
  if (status === 'error') return 'error'
  return 'n/a'
}

export function VerdictBanner({ report }: { report: AnalysisReport }) {
  const { verdict, categories } = report
  const metadata = categories.metadata_analysis

  return (
    <section className={`card verdict verdict-${verdict.label}`} aria-labelledby="verdict-title">
      <div className="verdict-head">
        <div className="verdict-text">
          <p className="eyebrow">Verdict</p>
          <h2 id="verdict-title">{VERDICT_LABELS[verdict.label]}</h2>
          <p className="muted file-line">
            <span className="mono">{report.file_name}</span> · {report.media_type}
            {metadata.checks_run !== undefined &&
              ` · ${metadata.checks_run} of ${metadata.checks_total} metadata checks could run`}
          </p>
        </div>
        <div className="score" aria-label={`Suspicion score ${verdict.score.toFixed(2)} out of 1`}>
          <span className="score-value">{verdict.score.toFixed(2)}</span>
          <span className="score-label">Suspicion score</span>
        </div>
      </div>

      <div className="meter" role="presentation">
        <div className="meter-fill" style={{ width: `${Math.round(verdict.score * 100)}%` }} />
      </div>

      <ul className="chips" aria-label="Results by category">
        {CATEGORY_ORDER.map((key) => {
          const category = categories[key]
          const flagged = category.checks.filter((c) => c.status === 'flagged').length
          return (
            <li key={key}>
              <a href={`#${key}`} className={`chip chip-${category.status}`}>
                <span className="chip-dot" aria-hidden="true" />
                {CATEGORY_SHORT[key]}
                <span className="chip-value">{chipText(category.status, flagged)}</span>
              </a>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
