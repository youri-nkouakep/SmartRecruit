import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { FileUp, ScanText, Trash2 } from 'lucide-react'
import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import { Badge, Card, Empty, ErrorNote, PageHeader, Spinner, pct } from '../components/ui'

const ACCEPT = '.pdf,.png,.jpg,.jpeg,.tif,.tiff'

export default function Candidates() {
  const qc = useQueryClient()
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const list = useQuery({
    queryKey: ['candidates'],
    queryFn: api.candidates,
    refetchInterval: (q) => (q.state.data?.some((c) => c.status === 'processing') ? 1500 : false),
  })
  const upload = useMutation({
    mutationFn: api.upload,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['candidates'] })
      qc.invalidateQueries({ queryKey: ['overview'] })
    },
  })
  const remove = useMutation({
    mutationFn: api.deleteCandidate,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['candidates'] }),
  })

  const send = (files: FileList | null) => files && files.length > 0 && upload.mutate(Array.from(files))

  return (
    <>
      <PageHeader
        title="Candidates"
        description="Digital PDFs are read from their text layer. Scans and photos are cleaned with OpenCV (lighting, skew, noise), then read by OCR. Layout analysis separates columns and sections."
      />

      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); send(e.dataTransfer.files) }}
        onClick={() => input.current?.click()}
        className={clsx(
          'flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition',
          dragging ? 'border-forest-500 bg-forest-50' : 'border-forest-200 bg-white hover:border-forest-400',
        )}
      >
        <div className="grid size-11 place-items-center rounded-full bg-forest-100 text-forest-700">
          {upload.isPending ? <Spinner /> : <FileUp className="size-5" />}
        </div>
        <p className="mt-3 text-sm font-medium">Drop resumes here, or click to browse</p>
        <p className="mt-1 text-xs text-muted">PDF, PNG, JPG or TIFF, up to 10 MB each. Several files at once is fine.</p>
        <input ref={input} type="file" multiple accept={ACCEPT} hidden onChange={(e) => send(e.target.files)} />
      </div>
      {upload.error && <div className="mt-3"><ErrorNote error={upload.error} /></div>}

      <Card className="mt-6 overflow-hidden">
        {list.isLoading ? (
          <div className="p-6"><Spinner /></div>
        ) : !list.data?.length ? (
          <div className="p-6"><Empty title="No candidates yet">Synthetic samples are in data/synthetic_resumes after running the benchmark.</Empty></div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-forest-50 text-left text-xs uppercase tracking-wide text-muted">
              <tr>
                <th className="px-5 py-3 font-medium">Candidate</th>
                <th className="px-5 py-3 font-medium">Predicted family</th>
                <th className="px-5 py-3 font-medium">Read with</th>
                <th className="px-5 py-3 font-medium text-right">Time</th>
                <th className="w-10" />
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {list.data.map((c) => (
                <tr key={c.id} className="hover:bg-forest-50/60">
                  <td className="px-5 py-3">
                    {c.status === 'ready' ? (
                      <Link to={`/candidates/${c.id}`} className="font-medium hover:text-forest-700">{c.resume?.name || c.filename}</Link>
                    ) : (
                      <span className="font-medium">{c.filename}</span>
                    )}
                    <p className="text-xs text-muted">{c.status === 'ready' ? c.resume?.headline : c.filename}</p>
                  </td>
                  <td className="px-5 py-3">
                    {c.status === 'processing' && <span className="flex items-center gap-2 text-muted"><Spinner className="size-3" />Reading</span>}
                    {c.status === 'error' && <Badge tone="clay">Failed: {c.error}</Badge>}
                    {c.prediction && (
                      <span className="flex items-center gap-2">
                        <Badge>{c.prediction.top[0].label}</Badge>
                        <span className="tabular text-xs text-muted">{pct(c.prediction.top[0].probability)}</span>
                      </span>
                    )}
                  </td>
                  <td className="px-5 py-3">
                    {c.source && (
                      <span className="inline-flex items-center gap-1.5 text-xs text-muted">
                        <ScanText className="size-3.5" />{c.source === 'OCR' ? 'OCR' : 'PDF text layer'}
                      </span>
                    )}
                  </td>
                  <td className="tabular px-5 py-3 text-right text-xs text-muted">{c.seconds ? `${c.seconds.toFixed(1)} s` : ''}</td>
                  <td className="pr-3">
                    <button onClick={() => remove.mutate(c.id)} className="rounded p-1.5 text-forest-300 hover:bg-clay/10 hover:text-clay" aria-label="Delete">
                      <Trash2 className="size-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  )
}
