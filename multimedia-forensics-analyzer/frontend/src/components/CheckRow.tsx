import type { Check } from '../api'
import { CHECK_STATUS_LABELS, formatValue, prettyCheck } from '../format'

export function CheckRow({ check }: { check: Check }) {
  const details = Object.entries(check.details)

  return (
    <li className={`check check-${check.status}`}>
      <div className="check-main">
        <div className="check-title">
          <span className="check-name">{check.name}</span>
          {check.status === 'flagged' && check.severity ? (
            <span className={`badge badge-${check.severity}`}>{check.severity}</span>
          ) : (
            <span className={`status status-${check.status}`}>{CHECK_STATUS_LABELS[check.status]}</span>
          )}
        </div>
        <p className="check-result">{check.result}</p>

        {details.length > 0 && (
          <details>
            <summary>Details</summary>
            <dl className="evidence">
              {details.map(([key, value]) => (
                <div key={key}>
                  <dt>{prettyCheck(key)}</dt>
                  <dd>{formatValue(value)}</dd>
                </div>
              ))}
            </dl>
          </details>
        )}
      </div>

      {check.visualization && (
        <figure className="check-visual">
          <img src={check.visualization} alt={`${check.name} visualization`} />
        </figure>
      )}
    </li>
  )
}
