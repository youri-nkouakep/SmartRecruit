import { lazy, Suspense } from 'react'
import { Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import { Spinner } from './components/ui'
import CandidateDetail from './pages/CandidateDetail'
import Candidates from './pages/Candidates'
import OfferDetail from './pages/OfferDetail'
import Offers from './pages/Offers'
import Overview from './pages/Overview'

const Fairness = lazy(() => import('./pages/Fairness'))
const Methodology = lazy(() => import('./pages/Methodology'))

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<Spinner />}>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/candidates" element={<Candidates />} />
        <Route path="/candidates/:id" element={<CandidateDetail />} />
        <Route path="/offers" element={<Offers />} />
        <Route path="/offers/:id" element={<OfferDetail />} />
        <Route path="/fairness" element={<Fairness />} />
        <Route path="/methodology" element={<Methodology />} />
      </Routes>
      </Suspense>
    </Layout>
  )
}
