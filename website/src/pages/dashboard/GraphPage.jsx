import { useOutletContext } from 'react-router-dom'
import GraphViz from '../../components/GraphViz'

export default function GraphPage() {
  const { userId } = useOutletContext()
  return <GraphViz userId={userId} />
}
