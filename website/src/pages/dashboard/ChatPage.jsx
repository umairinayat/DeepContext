import { useOutletContext } from 'react-router-dom'
import ChatPanel from '../../components/ChatPanel'

export default function ChatPage() {
  const { userId } = useOutletContext()
  return <ChatPanel userId={userId} />
}
