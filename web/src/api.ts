export interface Experience { title: string; company: string; period: string }
export interface Education { degree: string; school: string; year: string }
export interface Resume {
  name: string
  email: string
  phone: string
  headline: string
  summary: string
  skills: string[]
  experience: Experience[]
  education: Education[]
}
export interface Prediction {
  model: string
  top: { label: string; probability: number }[]
  key_terms: { term: string; weight: number }[]
}
export interface Line { text: string; box: [number, number, number, number]; section: string }
export interface Candidate {
  id: number
  filename: string
  status: 'processing' | 'ready' | 'error'
  error?: string
  source?: string
  seconds?: number
  resume?: Resume
  prediction?: Prediction
  lines?: Line[]
  created_at: number
}
export interface Offer { id: number; title: string; category: string | null; text: string; created_at: number }
export interface Evidence { requirement: string; evidence: string; similarity: number }
export interface Hit {
  ref: string
  title: string
  category: string | null
  score: number
  relevance: number
  evidence: Evidence[]
  candidate_id?: number
  name?: string
}
export interface Overview {
  offers: number
  candidates: number
  ready: number
  processing: number
  library: number
  families: Record<string, number>
  recent: Candidate[]
}
export type Row = Record<string, string | number>
export interface Benchmarks {
  docai: Row[]
  classification: Record<string, { accuracy: number; f1_macro: number; train_seconds: number; predict_ms: number }>
  retrieval: Row[]
  bias: Record<string, Record<string, number>>
  llm: Record<string, number | string>
  per_class: Row[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`)
  }
  return res.status === 204 ? (undefined as T) : res.json()
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  overview: () => request<Overview>('/overview'),
  benchmarks: () => request<Benchmarks>('/benchmarks'),
  offers: () => request<Offer[]>('/offers'),
  offer: (id: number) => request<Offer>(`/offers/${id}`),
  createOffer: (title: string, text: string) => request<Offer>('/offers', json('POST', { title, text })),
  deleteOffer: (id: number) => request<void>(`/offers/${id}`, { method: 'DELETE' }),
  candidates: () => request<Candidate[]>('/candidates'),
  candidate: (id: number) => request<Candidate>(`/candidates/${id}`),
  upload: (files: File[]) => {
    const form = new FormData()
    files.forEach((f) => form.append('files', f))
    return request<{ ids: number[] }>('/candidates', { method: 'POST', body: form })
  },
  updateResume: (id: number, resume: Resume) => request<Candidate>(`/candidates/${id}/resume`, json('PUT', resume)),
  deleteCandidate: (id: number) => request<void>(`/candidates/${id}`, { method: 'DELETE' }),
  match: (offerId: number, pool: 'uploaded' | 'library') => request<Hit[]>(`/offers/${offerId}/match?pool=${pool}`),
  imageUrl: (id: number) => `/api/candidates/${id}/image`,
}

export async function streamExplanation(
  offerId: number,
  evidence: Evidence[],
  onToken: (t: string) => void,
): Promise<{ cited_bullets: number; bullets: number }> {
  const res = await fetch('/api/explain', json('POST', { offer_id: offerId, evidence }))
  if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''
    for (const e of events) {
      const data = JSON.parse(e.replace(/^data: /, ''))
      if (data.error) throw new Error(data.error)
      if (data.done) return data
      onToken(data.token)
    }
  }
  throw new Error('Stream ended unexpectedly')
}
