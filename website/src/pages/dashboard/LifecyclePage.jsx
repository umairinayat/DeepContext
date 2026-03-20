import { useOutletContext } from 'react-router-dom'
import LifecycleControls from '../../components/LifecycleControls'

export default function LifecyclePage() {
  const { userId } = useOutletContext()
  return <LifecycleControls userId={userId} />
}
