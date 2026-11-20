import type { CheckStatus } from '../api'
import { prettyCheck } from '../format'

const LABELS: Record<CheckStatus, string> = {
  flagged: 'Flagged',
  not_flagged: 'Passed',
  not_applicable: 'Not enough data',
}

const ORDER: CheckStatus[] = ['flagged', 'not_flagged', 'not_applicable']

export function ChecksTable({ results }: { results: Record<string, CheckStatus> }) {
  const rows = Object.entries(results).sort(
    ([a, sa], [b, sb]) => ORDER.indexOf(sa) - ORDER.indexOf(sb) || a.localeCompare(b),
  )

  return (
    <section className="card">
      <h3>Metadata checks</h3>
      <table className="checks">
        <thead>
          <tr>
            <th>Check</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, status]) => (
            <tr key={name}>
              <td>{prettyCheck(name)}</td>
              <td>
                <span className={`status status-${status}`}>{LABELS[status]}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
