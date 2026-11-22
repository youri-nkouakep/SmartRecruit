import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Plus } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { Badge, Button, Card, CardHeader, ErrorNote, PageHeader, Spinner } from '../components/ui'

export default function Offers() {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const offers = useQuery({ queryKey: ['offers'], queryFn: api.offers })
  const [open, setOpen] = useState(false)
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const create = useMutation({
    mutationFn: () => api.createOffer(title, text),
    onSuccess: (o) => { qc.invalidateQueries({ queryKey: ['offers'] }); navigate(`/offers/${o.id}`) },
  })

  return (
    <>
      <PageHeader
        title="Job offers"
        description="Each offer is matched against your uploaded candidates or the public library of 2,483 resumes."
        action={<Button onClick={() => setOpen((v) => !v)}><Plus className="size-4" />New offer</Button>}
      />
      {open && (
        <Card className="mb-6">
          <CardHeader title="New offer" subtitle="Write the requirements as sentences: each one is matched to evidence in the resumes" />
          <form className="space-y-3 p-5" onSubmit={(e) => { e.preventDefault(); create.mutate() }}>
            <input className="w-full rounded-lg border border-line px-3 py-2 text-sm outline-none focus:border-forest-400" placeholder="Title, e.g. Data Analyst" value={title} onChange={(e) => setTitle(e.target.value)} />
            <textarea className="h-32 w-full rounded-lg border border-line px-3 py-2 text-sm outline-none focus:border-forest-400" placeholder="Missions and requirements" value={text} onChange={(e) => setText(e.target.value)} />
            {create.error && <ErrorNote error={create.error} />}
            <div className="flex justify-end gap-2">
              <Button type="button" variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
              <Button disabled={create.isPending || title.length < 3 || text.length < 30}>Create</Button>
            </div>
          </form>
        </Card>
      )}
      {offers.isLoading && <Spinner />}
      {offers.error && <ErrorNote error={offers.error} />}
      <div className="grid gap-4 md:grid-cols-2">
        {offers.data?.map((o) => (
          <Link key={o.id} to={`/offers/${o.id}`}>
            <Card className="group h-full p-5 transition hover:border-forest-300 hover:shadow-sm">
              <div className="flex items-start justify-between gap-3">
                <h3 className="font-medium text-forest-900">{o.title}</h3>
                <ArrowRight className="size-4 shrink-0 text-forest-300 transition group-hover:translate-x-0.5 group-hover:text-forest-600" />
              </div>
              <p className="mt-2 line-clamp-2 text-sm text-muted">{o.text.slice(o.title.length + 1)}</p>
              {o.category && <div className="mt-3"><Badge>{o.category}</Badge></div>}
            </Card>
          </Link>
        ))}
      </div>
    </>
  )
}
