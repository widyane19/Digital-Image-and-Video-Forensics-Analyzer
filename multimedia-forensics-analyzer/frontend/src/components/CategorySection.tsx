import type { Category, CategoryKey } from '../api'
import {
  CATEGORY_DESCRIPTIONS,
  CATEGORY_STATUS_LABELS,
  CATEGORY_TITLES,
} from '../format'
import { CheckRow } from './CheckRow'

export function CategorySection({ id, category }: { id: CategoryKey; category: Category }) {
  const shown = category.checks.filter((c) => c.status !== 'not_applicable')
  const notRun = category.checks.filter((c) => c.status === 'not_applicable')
  const inactive = category.status === 'not_applicable'

  return (
    <section
      id={id}
      className={`card category${inactive ? ' category-inactive' : ''}`}
      aria-labelledby={`${id}-title`}
    >
      <div className="section-head">
        <h3 id={`${id}-title`}>{CATEGORY_TITLES[id]}</h3>
        <span className={`pill pill-${category.status}`}>{CATEGORY_STATUS_LABELS[category.status]}</span>
      </div>
      <p className="muted small">{CATEGORY_DESCRIPTIONS[id]}</p>
      <p className="category-summary">{category.summary}</p>

      {shown.length > 0 && (
        <ul className="checks-list">
          {shown.map((check) => (
            <CheckRow key={check.id} check={check} />
          ))}
        </ul>
      )}

      {notRun.length > 0 && (
        <details className="not-run">
          <summary>
            {notRun.length} check{notRun.length === 1 ? '' : 's'} could not run
          </summary>
          <ul className="checks-list">
            {notRun.map((check) => (
              <CheckRow key={check.id} check={check} />
            ))}
          </ul>
        </details>
      )}
    </section>
  )
}
