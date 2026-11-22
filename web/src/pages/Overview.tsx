import { useQuery } from '@tanstack/react-query'
import { ArrowRight, FileUp, Plus } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { Badge, Button, Card, CardHeader, Empty, ErrorNote, Meter, PageHeader, Spinner, Stat, pct } from '../components/ui'

export default function Overview() {
  const overview = useQuery({ queryKey: ['overview'], queryFn: api.overview })
  const bench = useQuery({ queryKey: ['benchmarks'], queryFn: api.benchmarks })

  if (overview.error) return <ErrorNote error={overview.error} />
  const o = overview.data
  const b = bench.data
  const scan = b?.docai.find((r) => String(r.row).startsWith('Scan + OpenCV'))
  const finetuned = b && Object.entries(b.classification).find(([k]) => k.startsWith('Fine-tuned'))?.[1]
  const hybrid = b?.retrieval.find((r) => String(r.row).startsWith('Hybrid (BM25'))
  const families = Object.entries(o?.families ?? {}).sort((a, b) => b[1] - a[1])
  const maxFamily = Math.max(1, ...families.map(([, n]) => n))

  return (
    <>
      <PageHeader
        title="Overview"
        description="Read resumes from any PDF or scan, sort them by job family, and rank candidates for an offer with explanations grounded in each resume."
        action={
          <div className="flex gap-2">
            <Link to="/candidates"><Button variant="secondary"><FileUp className="size-4" />Upload resumes</Button></Link>
            <Link to="/offers"><Button><Plus className="size-4" />New offer</Button></Link>
          </div>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Open offers" value={o?.offers ?? '-'} />
        <Stat label="Uploaded candidates" value={o?.candidates ?? '-'} hint={o?.processing ? `${o.processing} being read` : undefined} />
        <Stat label="Public resume library" value={o ? o.library.toLocaleString('en') : '-'} hint="Searchable for every offer" />
        <Stat label="Job families" value="24" hint="Predicted by a fine-tuned DistilBERT" />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader title="Model quality" subtitle="Measured on held-out data, see Methodology" />
          {!b ? (
            <div className="p-6"><Spinner /></div>
          ) : (
            <ul className="divide-y divide-line">
              {[
                ['Field extraction on degraded scans', 'OCR + OpenCV preprocessing, F1', Number(scan?.overall ?? 0)],
                ['Job family classification', 'Fine-tuned DistilBERT, macro F1', finetuned?.f1_macro ?? 0],
                ['Relevant resumes in the top 10', 'Hybrid BM25 + dense search, P@10', Number(hybrid?.['P@10'] ?? 0)],
                ['LLM bullets citing resume evidence', 'Qwen2.5-1.5B, automatic check', Number(b.llm.citation_rate ?? 0)],
              ].map(([label, hint, value]) => (
                <li key={label as string} className="grid grid-cols-[1fr_auto] items-center gap-x-6 gap-y-2 px-5 py-4">
                  <div>
                    <p className="text-sm font-medium">{label}</p>
                    <p className="text-xs text-muted">{hint}</p>
                  </div>
                  <span className="tabular text-lg font-semibold text-forest-800">{pct(value as number)}</span>
                  <Meter value={value as number} className="col-span-2" />
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader title="Candidates by job family" subtitle="Uploaded resumes" />
          <div className="p-5">
            {families.length === 0 ? (
              <Empty title="No resumes yet">Upload a PDF or a photo of a resume to start.</Empty>
            ) : (
              <ul className="space-y-3">
                {families.slice(0, 8).map(([label, n]) => (
                  <li key={label}>
                    <div className="mb-1 flex justify-between text-sm"><span>{label}</span><span className="tabular text-muted">{n}</span></div>
                    <Meter value={n / maxFamily} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        </Card>
      </div>

      {o && o.recent.length > 0 && (
        <Card className="mt-6">
          <CardHeader title="Recently read" action={<Link to="/candidates" className="text-xs font-medium text-forest-700 hover:underline">See all</Link>} />
          <ul className="divide-y divide-line">
            {o.recent.map((c) => (
              <li key={c.id}>
                <Link to={`/candidates/${c.id}`} className="flex items-center justify-between px-5 py-3 hover:bg-forest-50">
                  <div>
                    <p className="text-sm font-medium">{c.resume?.name || c.filename}</p>
                    <p className="text-xs text-muted">{c.resume?.headline}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge>{c.prediction?.top[0].label}</Badge>
                    <ArrowRight className="size-4 text-forest-400" />
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </>
  )
}
