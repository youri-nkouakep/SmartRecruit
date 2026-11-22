import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { ArrowLeft, Plus, Save, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, type Line, type Resume } from '../api'
import { Badge, Button, Card, CardHeader, ErrorNote, Meter, Spinner, pct } from '../components/ui'

const SECTIONS: Record<string, { label: string; color: string }> = {
  header: { label: 'Identity', color: '#2a5c43' },
  heading: { label: 'Headings', color: '#b88a1b' },
  summary: { label: 'Summary', color: '#6a9680' },
  experience: { label: 'Experience', color: '#3c7558' },
  education: { label: 'Education', color: '#7a6a3a' },
  skills: { label: 'Skills', color: '#a5523f' },
  contact: { label: 'Contact', color: '#1f4632' },
}

function DocumentView({ id, lines }: { id: number; lines: Line[] }) {
  const present = Array.from(new Set(lines.map((l) => l.section))).filter((s) => SECTIONS[s])
  const [hidden, setHidden] = useState<string[]>([])
  const [hover, setHover] = useState<Line | null>(null)
  return (
    <Card className="overflow-hidden">
      <CardHeader title="What the model read" subtitle="Each box is a text line, colored by the section the layout analysis assigned it to" />
      <div className="flex flex-wrap gap-1.5 border-b border-line px-5 py-3">
        {present.map((s) => (
          <button
            key={s}
            onClick={() => setHidden((h) => (h.includes(s) ? h.filter((x) => x !== s) : [...h, s]))}
            className={clsx('flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs', hidden.includes(s) ? 'border-line text-muted opacity-50' : 'border-line')}
          >
            <span className="size-2.5 rounded-sm" style={{ background: SECTIONS[s].color }} />
            {SECTIONS[s].label}
          </button>
        ))}
      </div>
      <div className="relative bg-forest-50 p-3">
        <div className="relative">
          <img src={api.imageUrl(id)} alt="Resume" className="w-full rounded border border-line bg-white" />
          {lines.filter((l) => !hidden.includes(l.section) && SECTIONS[l.section]).map((l, i) => {
            const [x0, y0, x1, y1] = l.box
            const c = SECTIONS[l.section].color
            return (
              <div
                key={i}
                onMouseEnter={() => setHover(l)}
                onMouseLeave={() => setHover(null)}
                className="absolute rounded-[2px]"
                style={{ left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%`, border: `1.5px solid ${c}`, background: `${c}1f` }}
              />
            )
          })}
        </div>
        <div className="mt-3 min-h-10 rounded-md bg-white px-3 py-2 text-xs text-muted ring-1 ring-line">
          {hover ? <><span className="font-medium text-ink">{SECTIONS[hover.section]?.label}:</span> {hover.text}</> : 'Hover a box to see the text recognized in it.'}
        </div>
      </div>
    </Card>
  )
}

const input = 'w-full rounded-lg border border-line bg-white px-3 py-2 text-sm outline-none focus:border-forest-400 focus:ring-2 focus:ring-forest-100'

function Field({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-muted">{label}</span>
      <input className={input} value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  )
}

function ResumeForm({ value, onChange }: { value: Resume; onChange: (r: Resume) => void }) {
  const [skill, setSkill] = useState('')
  const set = <K extends keyof Resume>(k: K, v: Resume[K]) => onChange({ ...value, [k]: v })
  return (
    <div className="space-y-4 p-5">
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Name" value={value.name} onChange={(v) => set('name', v)} />
        <Field label="Headline" value={value.headline} onChange={(v) => set('headline', v)} />
        <Field label="Email" value={value.email} onChange={(v) => set('email', v)} />
        <Field label="Phone" value={value.phone} onChange={(v) => set('phone', v)} />
      </div>
      <div>
        <span className="mb-1 block text-xs font-medium text-muted">Skills</span>
        <div className="flex flex-wrap gap-1.5">
          {value.skills.map((s) => (
            <span key={s} className="inline-flex items-center gap-1 rounded-md bg-forest-50 px-2 py-1 text-xs ring-1 ring-forest-100">
              {s}
              <button onClick={() => set('skills', value.skills.filter((x) => x !== s))} aria-label={`Remove ${s}`}><X className="size-3 text-muted" /></button>
            </span>
          ))}
          <form onSubmit={(e) => { e.preventDefault(); if (skill.trim()) { set('skills', [...value.skills, skill.trim()]); setSkill('') } }}>
            <input className="w-28 rounded-md border border-dashed border-forest-200 px-2 py-1 text-xs outline-none focus:border-forest-400" placeholder="Add a skill" value={skill} onChange={(e) => setSkill(e.target.value)} />
          </form>
        </div>
      </div>
      {(['experience', 'education'] as const).map((key) => (
        <div key={key}>
          <div className="mb-1 flex items-center justify-between">
            <span className="text-xs font-medium text-muted">{key === 'experience' ? 'Experience' : 'Education'}</span>
            <button
              className="inline-flex items-center gap-1 text-xs text-forest-700 hover:underline"
              onClick={() => set(key, [...value[key], key === 'experience' ? { title: '', company: '', period: '' } : { degree: '', school: '', year: '' }] as never)}
            ><Plus className="size-3" />Add</button>
          </div>
          <div className="space-y-2">
            {value[key].map((row, i) => (
              <div key={i} className="grid grid-cols-[1fr_1fr_7rem_auto] gap-2">
                {Object.keys(row).map((f) => (
                  <input
                    key={f}
                    className={input}
                    placeholder={f}
                    value={(row as unknown as Record<string, string>)[f]}
                    onChange={(e) => set(key, value[key].map((r, j) => (j === i ? { ...r, [f]: e.target.value } : r)) as never)}
                  />
                ))}
                <button onClick={() => set(key, value[key].filter((_, j) => j !== i) as never)} className="px-1 text-forest-300 hover:text-clay" aria-label="Remove"><X className="size-4" /></button>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

export default function CandidateDetail() {
  const id = Number(useParams().id)
  const qc = useQueryClient()
  const q = useQuery({ queryKey: ['candidate', id], queryFn: () => api.candidate(id) })
  const [draft, setDraft] = useState<Resume | null>(null)
  useEffect(() => { if (q.data?.resume) setDraft(q.data.resume) }, [q.data])
  const save = useMutation({
    mutationFn: (r: Resume) => api.updateResume(id, r),
    onSuccess: (c) => { qc.setQueryData(['candidate', id], c); qc.invalidateQueries({ queryKey: ['candidates'] }) },
  })

  if (q.error) return <ErrorNote error={q.error} />
  if (!q.data || !draft) return <Spinner />
  const c = q.data
  const dirty = JSON.stringify(draft) !== JSON.stringify(c.resume)

  return (
    <>
      <Link to="/candidates" className="mb-4 inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink"><ArrowLeft className="size-4" />Candidates</Link>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-forest-900">{c.resume?.name || c.filename}</h1>
          <p className="mt-1 text-sm text-muted">
            {c.filename} · read with {c.source === 'OCR' ? 'OCR after OpenCV preprocessing' : 'the PDF text layer'} in {c.seconds?.toFixed(1)} s
          </p>
        </div>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <DocumentView id={id} lines={c.lines ?? []} />
        <div className="space-y-6">
          {c.prediction && (
            <Card>
              <CardHeader title="Predicted job family" subtitle={c.prediction.model} />
              <ul className="space-y-3 px-5 py-4">
                {c.prediction.top.map((p, i) => (
                  <li key={p.label}>
                    <div className="mb-1 flex justify-between text-sm">
                      <span className={clsx(i === 0 && 'font-medium')}>{p.label}</span>
                      <span className="tabular text-muted">{pct(p.probability)}</span>
                    </div>
                    <Meter value={p.probability} />
                  </li>
                ))}
              </ul>
              <div className="border-t border-line px-5 py-3">
                <p className="mb-2 text-xs text-muted">Terms that weigh most for this prediction (linear model, exact contributions)</p>
                <div className="flex flex-wrap gap-1.5">{c.prediction.key_terms.map((t) => <Badge key={t.term}>{t.term}</Badge>)}</div>
              </div>
            </Card>
          )}
          <Card>
            <CardHeader
              title="Extracted profile"
              subtitle="Correct any field, then save: the prediction is recomputed"
              action={<Button disabled={!dirty || save.isPending} onClick={() => save.mutate(draft)}>{save.isPending ? <Spinner className="size-3" /> : <Save className="size-4" />}Save</Button>}
            />
            <ResumeForm value={draft} onChange={setDraft} />
            {save.error && <div className="px-5 pb-5"><ErrorNote error={save.error} /></div>}
          </Card>
        </div>
      </div>
    </>
  )
}
