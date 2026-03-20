import { useOutletContext } from 'react-router-dom'
import StatsCards from '../../components/StatsCards'

export default function StatsPage() {
  const { userId } = useOutletContext()
  return <StatsCards userId={userId} />
}
