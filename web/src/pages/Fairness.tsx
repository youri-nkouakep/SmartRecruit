import { useQuery } from '@tanstack/react-query'
import { ShieldCheck } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import { Card, CardHeader, ErrorNote, PageHeader, Spinner } from '../components/ui'

const GROUPS = [
  ['gender:female', 'Women'],
  ['gender:male', 'Men'],
  ['origin:white', 'White-associated names'],
  ['origin:black', 'Black-associated names'],
  ['origin:north_african', 'North African names'],
] as const

export default function Fairness() {
  const q = useQuery({ queryKey: ['benchmarks'], queryFn: api.benchmarks })
  if (q.error) return <ErrorNote error={q.error} />
  if (!q.data) return <Spinner />
  const bias = q.data.bias
  const cross = bias['Cross-encoder score'] ?? {}
  const chart = GROUPS.map(([k, label]) => ({ label, value: cross[k] ?? 0 }))

  return (
    <>
      <PageHeader
        title="Fairness"
        description="Recruitment is a high-risk use of AI under the EU AI Act. Every model was audited with counterfactual resumes: the same resume, only the candidate's name and pronoun change."
      />
      <Card className="mb-6 flex gap-4 border-forest-200 bg-forest-50 px-5 py-4">
        <ShieldCheck className="mt-0.5 size-5 shrink-0 text-forest-700" />
        <div className="text-sm">
          <p className="font-medium text-forest-900">Identity masking is built in</p>
          <p className="mt-1 text-muted">
            Ranking and classification only see the structured profile: headline, summary, skills, experience and education.
            The name, email and phone number extracted from the resume are kept for the recruiter but never sent to the models.
          </p>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader title="Audit results" subtitle="100 test resumes x 18 names (2 genders x 3 origins x 3 names)" />
          <table className="w-full text-sm">
            <thead className="bg-forest-50 text-left text-xs uppercase tracking-wide text-muted">
              <tr><th className="px-5 py-2.5 font-medium">Model</th><th className="px-3 py-2.5 text-right font-medium">Decision changed</th><th className="px-3 py-2.5 text-right font-medium">Gender gap</th><th className="px-5 py-2.5 text-right font-medium">Origin gap</th></tr>
            </thead>
            <tbody className="tabular divide-y divide-line">
              {Object.entries(bias).map(([model, r]) => (
                <tr key={model}>
                  <td className="px-5 py-3">{model}</td>
                  <td className="px-3 py-3 text-right">{r.decision_flip_rate !== undefined ? `${(r.decision_flip_rate * 100).toFixed(1)} %` : 'n/a'}</td>
                  <td className="px-3 py-3 text-right">{r.gender_gap.toFixed(4)}</td>
                  <td className="px-5 py-3 text-right">{r.origin_gap.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="space-y-2 border-t border-line px-5 py-4 text-sm text-muted">
            <p>The linear TF-IDF model is almost blind to names. The fine-tuned transformer changes its prediction in 2.6 % of cases because of the name alone.</p>
            <p>This is why the pipeline removes identity fields before scoring instead of trusting any single model to ignore them.</p>
          </div>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader title="Cross-encoder score shift" subtitle="Mean change of the relevance logit when a name is added. Lower means the candidate is ranked lower." />
          <div className="h-72 px-2 py-4">
            <ResponsiveContainer>
              <BarChart data={chart} layout="vertical" margin={{ left: 20, right: 16 }}>
                <CartesianGrid horizontal={false} stroke="#e6efe8" />
                <XAxis type="number" domain={[(min: number) => Math.floor(min * 20) / 20, 0]} tick={{ fontSize: 11 }} stroke="#9ebdac" />
                <YAxis type="category" dataKey="label" width={130} tick={{ fontSize: 11 }} stroke="#9ebdac" />
                <ReferenceLine x={0} stroke="#5d6b63" />
                <Tooltip formatter={(v) => Number(v).toFixed(3)} />
                <Bar dataKey="value" fill="#2a5c43" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </>
  )
}
