import { useOutletContext } from 'react-router-dom'
import MemoryBrowser from '../../components/MemoryBrowser'

export default function MemoriesPage() {
  const { userId } = useOutletContext()
  return <MemoryBrowser userId={userId} />
}
