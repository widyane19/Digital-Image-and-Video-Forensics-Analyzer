import type { CategoryKey, CategoryStatus, CheckStatus, VerdictLabel } from './api'

// Turn a check or evidence name like 'modified_after_capture' into readable text.
export const prettyCheck = (name: string) => name.replace(/_/g, ' ')

export const CATEGORY_ORDER: CategoryKey[] = [
  'metadata_analysis',
  'image_analysis',
  'compression_analysis',
  'video_analysis',
]

export const CATEGORY_TITLES: Record<CategoryKey, string> = {
  metadata_analysis: 'Metadata analysis',
  image_analysis: 'Image analysis',
  compression_analysis: 'Compression analysis',
  video_analysis: 'Video analysis',
}

export const CATEGORY_SHORT: Record<CategoryKey, string> = {
  metadata_analysis: 'Metadata',
  image_analysis: 'Image',
  compression_analysis: 'Compression',
  video_analysis: 'Video',
}

export const CATEGORY_DESCRIPTIONS: Record<CategoryKey, string> = {
  metadata_analysis: 'Dates, software and other information stored inside the file.',
  image_analysis: 'The pixels themselves: duplicated regions, edges and texture.',
  compression_analysis: 'How the file was compressed and whether it was saved more than once.',
  video_analysis: 'Frame timing, repeated or inserted footage, and the audio and video tracks.',
}

export const CATEGORY_STATUS_LABELS: Record<CategoryStatus, string> = {
  flagged: 'Flagged',
  clear: 'No flags',
  not_applicable: 'Not applicable',
  error: 'Could not run',
}

export const CHECK_STATUS_LABELS: Record<CheckStatus, string> = {
  flagged: 'Flagged',
  passed: 'Passed',
  info: 'Measured',
  not_applicable: 'Not run',
}

export const VERDICT_LABELS: Record<VerdictLabel, string> = {
  likely_altered: 'Likely altered',
  some_signs: 'Some signs of alteration',
  no_strong_evidence: 'No strong evidence of alteration',
  inconclusive: 'Inconclusive',
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (Array.isArray(value)) return value.map(formatValue).join(', ')
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : String(value)
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
