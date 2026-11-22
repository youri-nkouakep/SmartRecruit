import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { ArrowLeft, ChevronDown, Sparkles } from 'lucide-react'
import { Fragment, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, streamExplanation, type Hit } from '../api'
import { Badge, Button, Card, Empty, ErrorNote, Meter, Spinner, pct } from '../components/ui'

function Cited({ text }: { text: string }) {
  const parts = text.split(/(\[E\d+\]|\*\*[^*]+\*\*)/g)
  return (
    <>
      {parts.map((p, i) => {
        if (/^\[E\d+\]$/.test(p)) return <span key={i} className="mx-0.5 rounded bg-forest-100 px-1 py-px text-[11px] font-semibold text-forest-700">{p.slice(1, -1)}</span>
        if (/^\*\*.+\*\*$/.test(p)) return <strong key={i} className="font-semibold">{p.slice(2, -2)}</strong>
        return <Fragment key={i}>{p}</Fragment>
      })}
    </>
  )
}

function Explanation({ offerId, hit }: { offerId: number; hit: Hit }) {
  const [text, setText] = useState('')
  const [state, setState] = useState<'idle' | 'running' | 'done' | 'error'>('idle')
  const [check, setCheck] = useState<{ cited_bullets: number; bullets: number } | null>(null)
  const [error, setError] = useState<unknown>(null)
  const run = async () => {
    setText(''); setState('running'); setCheck(null)
    try {
      setCheck(await streamExplanation(offerId, hit.evidence, (t) => setText((s) => s + t)))
      setState('done')
    } catch (e) { setError(e); setState('error') }
  }
  return (
    <div className="border-t border-line bg-forest-50/60 px-5 py-4">
      {state === 'idle' ? (
        <Button variant="secondary" onClick={run}><Sparkles className="size-4" />Explain this match</Button>
      ) : (
        <div>
          <div className="mb-2 flex items-center gap-2 text-xs text-muted">
            <Sparkles className="size-3.5 text-forest-600" />Local LLM, constrained to the evidence above. The first run loads the model.
            {state === 'running' && <Spinner className="size-3" />}
            {check && (
              <Badge tone={check.cited_bullets === check.bullets ? 'green' : 'amber'}>
                {check.cited_bullets}/{check.bullets} bullets cite valid evidence
              </Badge>
            )}
          </div>
          {state === 'error' ? <ErrorNote error={error} /> : (
            <div className="whitespace-pre-wrap text-sm leading-relaxed"><Cited text={text || ' '} /></div>
          )}
        </div>
      )}
    </div>
  )
}

function HitCard({ hit, rank, offerId }: { hit: Hit; rank: number; offerId: number }) {
  const [open, setOpen] = useState(rank === 1)
  const title = hit.name ? (
    <Link to={`/candidates/${hit.candidate_id}`} className="hover:text-forest-700">{hit.name}</Link>
  ) : `Resume ${hit.ref.replace('library-', '#')}`
  return (
    <Card className="overflow-hidden">
      <div className="grid grid-cols-[2.5rem_1fr_9rem_2rem] items-center gap-4 px-5 py-4">
        <span className="tabular text-lg font-semibold text-forest-300">{rank}</span>
        <div className="min-w-0">
          <p className="truncate font-medium">{title}</p>
          <div className="mt-1 flex items-center gap-2 text-xs text-muted">
            {hit.category && <Badge>{hit.category}</Badge>}
            {hit.name && <span className="truncate">{hit.title}</span>}
          </div>
        </div>
        <div>
          <div className="mb-1 flex justify-between text-xs"><span className="text-muted">Relevance</span><span className="tabular font-medium">{pct(hit.relevance)}</span></div>
          <Meter value={hit.relevance} />
        </div>
        <button onClick={() => setOpen((v) => !v)} className="text-forest-400 hover:text-forest-700" aria-label="Details">
          <ChevronDown className={clsx('size-5 transition', open && 'rotate-180')} />
        </button>
      </div>
      {open && (
        <>
          <table className="w-full border-t border-line text-sm">
            <thead className="bg-forest-50 text-left text-[11px] uppercase tracking-wide text-muted">
              <tr><th className="w-12 px-5 py-2 font-medium">Id</th><th className="px-2 py-2 font-medium">Requirement</th><th className="px-2 py-2 font-medium">Closest sentence in the resume</th><th className="px-5 py-2 text-right font-medium">Similarity</th></tr>
            </thead>
            <tbody className="divide-y divide-line">
              {hit.evidence.map((e, i) => (
                <tr key={i} className="align-top">
                  <td className="px-5 py-2.5 text-xs font-semibold text-forest-600">E{i + 1}</td>
                  <td className="px-2 py-2.5 text-xs">{e.requirement}</td>
                  <td className="px-2 py-2.5 text-xs text-muted">{e.evidence.length > 220 ? `${e.evidence.slice(0, 220)}...` : e.evidence}</td>
                  <td className="tabular px-5 py-2.5 text-right text-xs">{e.similarity.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Explanation offerId={offerId} hit={hit} />
        </>
      )}
    </Card>
  )
}

export default function OfferDetail() {
  const id = Number(useParams().id)
  const offer = useQuery({ queryKey: ['offer', id], queryFn: () => api.offer(id) })
  const [pool, setPool] = useState<'uploaded' | 'library'>('uploaded')
  const [run, setRun] = useState(false)
  const hits = useQuery({ queryKey: ['match', id, pool], queryFn: () => api.match(id, pool), enabled: run })

  if (offer.error) return <ErrorNote error={offer.error} />
  if (!offer.data) return <Spinner />
  return (
    <>
      <Link to="/offers" className="mb-4 inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink"><ArrowLeft className="size-4" />Job offers</Link>
      <h1 className="text-2xl font-semibold tracking-tight text-forest-900">{offer.data.title}</h1>
      <Card className="mt-4">
        <p className="px-5 py-4 text-sm leading-relaxed text-ink/80">{offer.data.text}</p>
      </Card>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex rounded-lg border border-line bg-white p-1 text-sm">
          {(['uploaded', 'library'] as const).map((p) => (
            <button key={p} onClick={() => setPool(p)} className={clsx('rounded-md px-3 py-1.5', pool === p ? 'bg-forest-700 text-white' : 'text-muted hover:text-ink')}>
              {p === 'uploaded' ? 'My candidates' : 'Public library (2,483)'}
            </button>
          ))}
        </div>
        <Button onClick={() => { setRun(true); hits.refetch() }} disabled={hits.isFetching}>
          {hits.isFetching ? <Spinner className="size-3" /> : null}Rank candidates
        </Button>
      </div>
      <p className="mt-2 text-xs text-muted">BM25 and dense retrieval (bge-small, FAISS) are fused, then a cross-encoder re-ranks the top 50. Identity fields are excluded from the text the models see.</p>

      <div className="mt-5 space-y-3">
        {hits.error && <ErrorNote error={hits.error} />}
        {hits.isFetching && <Card className="flex items-center gap-3 px-5 py-6 text-sm text-muted"><Spinner />Searching and re-ranking...</Card>}
        {!hits.isFetching && hits.data?.length === 0 && <Empty title="No uploaded candidates">Upload resumes first, or search the public library.</Empty>}
        {!hits.isFetching && hits.data?.map((h, i) => <HitCard key={h.ref} hit={h} rank={i + 1} offerId={id} />)}
      </div>
    </>
  )
}
