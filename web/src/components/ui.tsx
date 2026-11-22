import clsx from 'clsx'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={clsx('rounded-xl border border-line bg-white', className)}>{children}</div>
}

export function CardHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
      <div>
        <h2 className="text-sm font-semibold text-ink">{title}</h2>
        {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
      </div>
      {action}
    </div>
  )
}

export function PageHeader({ title, description, action }: { title: string; description?: string; action?: ReactNode }) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-forest-900">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-muted">{description}</p>}
      </div>
      {action}
    </div>
  )
}

type Variant = 'primary' | 'secondary' | 'ghost'

export function Button({ variant = 'primary', className, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...props}
      className={clsx(
        'inline-flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50',
        variant === 'primary' && 'bg-forest-700 text-white hover:bg-forest-800',
        variant === 'secondary' && 'border border-line bg-white text-ink hover:border-forest-300 hover:bg-forest-50',
        variant === 'ghost' && 'text-muted hover:bg-forest-50 hover:text-ink',
        className,
      )}
    />
  )
}

export function Badge({ tone = 'neutral', children }: { tone?: 'neutral' | 'green' | 'amber' | 'clay'; children: ReactNode }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium',
        tone === 'neutral' && 'bg-forest-50 text-forest-700 ring-1 ring-forest-100',
        tone === 'green' && 'bg-forest-100 text-forest-800',
        tone === 'amber' && 'bg-amber/10 text-amber',
        tone === 'clay' && 'bg-clay/10 text-clay',
      )}
    >
      {children}
    </span>
  )
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <Card className="px-5 py-4">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">{label}</p>
      <p className="tabular mt-2 text-2xl font-semibold text-forest-900">{value}</p>
      {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </Card>
  )
}

export function Meter({ value, className }: { value: number; className?: string }) {
  return (
    <div className={clsx('h-1.5 w-full overflow-hidden rounded-full bg-forest-100', className)}>
      <div className="h-full rounded-full bg-forest-600" style={{ width: `${Math.max(2, Math.min(100, value * 100))}%` }} />
    </div>
  )
}

export function Spinner({ className }: { className?: string }) {
  return <span className={clsx('inline-block size-4 animate-spin rounded-full border-2 border-forest-200 border-t-forest-600', className)} />
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-forest-200 bg-white/60 px-6 py-12 text-center">
      <p className="text-sm font-medium text-ink">{title}</p>
      {children && <div className="mt-2 text-sm text-muted">{children}</div>}
    </div>
  )
}

export function ErrorNote({ error }: { error: unknown }) {
  return (
    <div className="rounded-lg border border-clay/30 bg-clay/5 px-4 py-3 text-sm text-clay">
      {error instanceof Error ? error.message : 'Something went wrong'}
    </div>
  )
}

export const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)} %`
