import type { Finding } from '../api'
import { prettyCheck } from '../format'

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function FindingsList({ findings }: { findings: Finding[] }) {
  return (
    <section className="card">
      <h3>Findings</h3>
      {findings.length === 0 ? (
        <p className="muted">No signs of editing were found in the metadata.</p>
      ) : (
        <ul className="findings">
          {findings.map((finding, i) => (
            <li key={`${finding.check}-${i}`} className="finding">
              <span className={`badge badge-${finding.severity}`}>{finding.severity}</span>
              <div>
                <p className="finding-check">{prettyCheck(finding.check)}</p>
                <p>{finding.message}</p>
                {Object.keys(finding.evidence).length > 0 && (
                  <details>
                    <summary>Evidence</summary>
                    <dl className="evidence">
                      {Object.entries(finding.evidence).map(([key, value]) => (
                        <div key={key}>
                          <dt>{prettyCheck(key)}</dt>
                          <dd>{formatValue(value)}</dd>
                        </div>
                      ))}
                    </dl>
                  </details>
                )}
                {finding.frames.length > 0 && (
                  <p className="muted small">Frames: {finding.frames.join(', ')}</p>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
