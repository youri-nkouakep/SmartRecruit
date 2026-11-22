import clsx from 'clsx'
import { BriefcaseBusiness, FlaskConical, LayoutDashboard, Scale, Users } from 'lucide-react'
import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/candidates', label: 'Candidates', icon: Users },
  { to: '/offers', label: 'Job offers', icon: BriefcaseBusiness },
  { to: '/fairness', label: 'Fairness', icon: Scale },
  { to: '/methodology', label: 'Methodology', icon: FlaskConical },
]

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid size-8 place-items-center rounded-lg bg-forest-700 text-sm font-bold text-forest-100">S</div>
      <div>
        <p className="text-sm font-semibold tracking-tight text-white">SmartRecruit</p>
        <p className="text-[11px] text-forest-300">AI resume screening</p>
      </div>
    </div>
  )
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col bg-forest-950 px-4 py-6 md:flex">
        <Logo />
        <nav className="mt-10 flex flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition',
                  isActive ? 'bg-forest-800 text-white' : 'text-forest-300 hover:bg-forest-900 hover:text-white',
                )
              }
            >
              <Icon className="size-4" strokeWidth={1.8} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto rounded-lg border border-forest-800 px-3 py-3 text-[11px] leading-relaxed text-forest-300">
          All models run locally. Names, emails and phone numbers are never sent to the ranking models.
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-6 py-8 md:px-10 lg:px-14">
        <div className="mx-auto max-w-6xl">{children}</div>
      </main>
    </div>
  )
}
