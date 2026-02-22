import { useEffect, useRef } from 'react'
import Prism from 'prismjs'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-json'

export default function CodeBlock({ code, language = 'python', header }) {
    const codeRef = useRef()

    useEffect(() => {
        if (codeRef.current) {
            Prism.highlightElement(codeRef.current)
        }
    }, [code])

    return (
        <div className="code-preview">
            {header !== false && (
                <div className="code-header">
                    <span className="dot red"></span>
                    <span className="dot yellow"></span>
                    <span className="dot green"></span>
                    <span style={{ marginLeft: '0.5rem' }}>{header || `${language}`}</span>
                </div>
            )}
            <pre style={header !== false ? { borderTopLeftRadius: 0, borderTopRightRadius: 0 } : {}}>
                <code ref={codeRef} className={`language-${language}`}>
                    {code.trim()}
                </code>
            </pre>
        </div>
    )
}
