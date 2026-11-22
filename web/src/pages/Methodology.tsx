import { useQuery } from '@tanstack/react-query'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type Row } from '../api'
import { Card, CardHeader, ErrorNote, PageHeader, Spinner } from '../components/ui'

const STEPS = [
  ['Read', 'PDF text layer, or OpenCV cleaning + Tesseract OCR for scans'],
  ['Structure', 'Column detection, section segmentation, field extraction'],
  ['Classify', 'Fine-tuned DistilBERT over 24 job families'],
  ['Retrieve', 'BM25 + bge-small dense search (FAISS), fused with RRF'],
  ['Re-rank', 'Cross-encoder on the top 50'],
  ['Explain', 'Evidence per requirement, then a local LLM with citation checks'],
]

function Table({ rows, columns, digits = 3 }: { rows: Row[]; columns: [string, string][]; digits?: number }) {
  return (
    <table className="w-full text-sm">
      <thead className="bg-forest-50 text-left text-xs uppercase tracking-wide text-muted">
        <tr>{columns.map(([k, l], i) => <th key={k} className={`px-5 py-2.5 font-medium ${i ? 'text-right' : ''}`}>{l}</th>)}</tr>
      </thead>
      <tbody className="tabular divide-y divide-line">
        {rows.map((r, i) => (
          <tr key={i}>{columns.map(([k], j) => <td key={k} className={`px-5 py-2.5 ${j ? 'text-right' : ''}`}>{j ? Number(r[k]).toFixed(digits) : r[k]}</td>)}</tr>
        ))}
      </tbody>
    </table>
  )
}

export default function Methodology() {
  const q = useQuery({ queryKey: ['benchmarks'], queryFn: api.benchmarks })
  if (q.error) return <ErrorNote error={q.error} />
  if (!q.data) return <Spinner />
  const b = q.data
  const short = (n: string) => (n.startsWith('TF-IDF') ? 'TF-IDF' : n.startsWith('Embeddings') ? 'bge-small embeddings' : 'DistilBERT fine-tuned')
  const order = ['TF-IDF', 'bge-small embeddings', 'DistilBERT fine-tuned']
  const cls = Object.entries(b.classification).map(([name, r]) => ({ ...r, name: short(name) })).sort((x, y) => order.indexOf(x.name) - order.indexOf(y.name))
  const doc = b.docai.map((r) => ({ name: String(r.row).replace(' + OpenCV preprocessing', ' + OpenCV'), single: Number(r.overall_single), two: Number(r.overall_two_column) }))

  return (
    <>
      <PageHeader title="Methodology" description="Every number in the app comes from scripts in the repository, run on held-out data." />
      <ol className="mb-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {STEPS.map(([t, d], i) => (
          <li key={t} className="rounded-xl border border-line bg-white px-4 py-3">
            <p className="text-xs font-semibold text-forest-500">Step {i + 1}</p>
            <p className="mt-0.5 font-medium">{t}</p>
            <p className="mt-1 text-xs text-muted">{d}</p>
          </li>
        ))}
      </ol>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Document AI" subtitle="Field extraction F1 on 50 synthetic resumes, digital and scanned" />
          <div className="h-64 px-2 py-4">
            <ResponsiveContainer>
              <BarChart data={doc} layout="vertical" margin={{ left: 30, right: 16 }}>
                <CartesianGrid horizontal={false} stroke="#e6efe8" />
                <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#9ebdac" />
                <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11 }} stroke="#9ebdac" />
                <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="single" name="Single column" fill="#3c7558" radius={[0, 3, 3, 0]} />
                <Bar dataKey="two" name="Two columns" fill="#b88a1b" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card>
          <CardHeader title="Job family classification" subtitle="497 test resumes, job titles removed to avoid leakage" />
          <div className="h-64 px-2 py-4">
            <ResponsiveContainer>
              <BarChart data={cls} margin={{ right: 16 }}>
                <CartesianGrid vertical={false} stroke="#e6efe8" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} stroke="#9ebdac" />
                <YAxis domain={[0, 0.8]} tick={{ fontSize: 11 }} stroke="#9ebdac" />
                <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="accuracy" name="Accuracy" fill="#9ebdac" radius={[3, 3, 0, 0]} />
                <Bar dataKey="f1_macro" name="Macro F1" fill="#1f4632" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card>
          <CardHeader title="Offer to resume retrieval" subtitle="24 offers, 2,483 resumes" />
          <Table rows={b.retrieval} columns={[['row', 'Method'], ['P@10', 'P@10'], ['nDCG@10', 'nDCG@10'], ['MRR', 'MRR']]} />
        </Card>
        <Card>
          <CardHeader title="LLM explanations" subtitle={String(b.llm.model ?? '')} />
          <dl className="tabular grid grid-cols-2 gap-px bg-line">
            {[
              ['Offers explained', b.llm.offers],
              ['Bullets citing valid evidence', `${b.llm.cited_bullets} / ${b.llm.bullets}`],
              ['Fully cited answers', `${b.llm.answers_fully_cited} / ${b.llm.offers}`],
              ['Mean generation time (CPU)', `${Number(b.llm.mean_seconds).toFixed(0)} s`],
            ].map(([k, v]) => (
              <div key={k as string} className="bg-white px-5 py-4">
                <dt className="text-xs text-muted">{k}</dt>
                <dd className="mt-1 text-lg font-semibold text-forest-900">{v}</dd>
              </div>
            ))}
          </dl>
        </Card>
      </div>
    </>
  )
}
