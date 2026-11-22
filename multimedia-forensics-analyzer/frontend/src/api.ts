// Types mirror the JSON built by src/pipeline.py and src/ai_summary.py.

export type Severity = 'info' | 'low' | 'medium' | 'high'
export type CheckStatus = 'flagged' | 'passed' | 'not_applicable' | 'info'
export type CategoryStatus = 'flagged' | 'clear' | 'not_applicable' | 'error'
export type VerdictLabel = 'likely_altered' | 'some_signs' | 'no_strong_evidence' | 'inconclusive'
export type CategoryKey =
  | 'metadata_analysis'
  | 'image_analysis'
  | 'compression_analysis'
  | 'video_analysis'

export interface Check {
  id: string
  name: string
  status: CheckStatus
  severity: Severity | null
  result: string
  details: Record<string, unknown>
  visualization: string | null
}

export interface Category {
  status: CategoryStatus
  summary: string
  checks: Check[]
  // metadata_analysis only
  score?: number
  checks_run?: number
  checks_total?: number
}

export interface AnalysisReport {
  file_name: string
  media_type: 'image' | 'video'
  verdict: { label: VerdictLabel; score: number }
  categories: Record<CategoryKey, Category>
  warnings: string[]
}

export interface Summary {
  text: string
  reasons: { check: string; category: string; severity: Severity | null; result: string }[]
  model: string | null
  generated: boolean
  note: string | null
}

async function readError(response: Response): Promise<Error> {
  let detail = `${response.status} ${response.statusText}`
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') detail = body.detail
  } catch {
    // The response was not JSON; keep the status text.
  }
  return new Error(detail)
}

export async function analyzeFile(file: File, signal?: AbortSignal): Promise<AnalysisReport> {
  const form = new FormData()
  form.append('file', file)
  // Saving the upload on the server gives it a new date, so send the original one.
  form.append('last_modified', String(file.lastModified))

  const response = await fetch('/api/analyze', { method: 'POST', body: form, signal })
  if (!response.ok) throw await readError(response)
  return response.json()
}

export async function summarize(report: AnalysisReport, signal?: AbortSignal): Promise<Summary> {
  // The summary only needs the results, so leave the evidence images out.
  const categories = Object.fromEntries(
    Object.entries(report.categories).map(([key, category]) => [
      key,
      { ...category, checks: category.checks.map((check) => ({ ...check, visualization: null })) },
    ]),
  )
  const response = await fetch('/api/summary', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...report, categories }),
    signal,
  })
  if (!response.ok) throw await readError(response)
  return response.json()
}
