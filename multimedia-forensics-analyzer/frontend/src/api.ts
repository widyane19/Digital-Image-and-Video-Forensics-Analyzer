// Types mirror the JSON built by src/pipeline.py and MetadataReport.to_dict().

export type Severity = 'info' | 'low' | 'medium' | 'high'
export type CheckStatus = 'flagged' | 'not_flagged' | 'not_applicable'
export type Box = [number, number, number, number] // x, y, width, height

export interface Finding {
  check: string
  severity: Severity
  message: string
  evidence: Record<string, unknown>
  regions: Box[]
  frames: number[]
}

export interface MetadataReport {
  path: string
  media_type: 'image' | 'video'
  score: number
  conclusion: string
  checks_run: number
  checks_total: number
  findings: Finding[]
  results: Record<string, CheckStatus>
  warnings: string[]
}

export interface ImageReport {
  copy_move?: {
    match_count: number
    candidate_regions: { source: Box; destination: Box; match_count: number }[]
    visualization: string
  }
  compression?: {
    file_size_bytes: number
    blockiness_ratio: number
    boundary_mean_difference: number
    interior_mean_difference: number
    visualization: string
  }
  edge_texture?: {
    edge_density: number
    laplacian_variance: number
    edge_map: string
  }
}

export interface AnalysisReport {
  file_name: string
  media_type: 'image' | 'video'
  metadata: MetadataReport
  image?: ImageReport
  video_detectors?: string
  warnings: string[]
}

export async function analyzeFile(file: File, signal?: AbortSignal): Promise<AnalysisReport> {
  const form = new FormData()
  form.append('file', file)
  // Saving the upload on the server gives it a new date, so send the original one.
  form.append('last_modified', String(file.lastModified))

  const response = await fetch('/api/analyze', { method: 'POST', body: form, signal })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // The response was not JSON; keep the status text.
    }
    throw new Error(detail)
  }
  return response.json()
}
