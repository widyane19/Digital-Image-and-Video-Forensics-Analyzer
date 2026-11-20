import { useRef, useState, type DragEvent } from 'react'

interface Props {
  onFile: (file: File) => void
  disabled?: boolean
}

export function UploadDropzone({ onFile, disabled }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function handleDrop(event: DragEvent) {
    event.preventDefault()
    setDragging(false)
    const file = event.dataTransfer.files[0]
    if (file && !disabled) onFile(file)
  }

  return (
    <div
      className={`dropzone${dragging ? ' dragging' : ''}${disabled ? ' disabled' : ''}`}
      role="button"
      tabIndex={0}
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={input}
        type="file"
        accept="image/*,video/*"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = '' // allow choosing the same file again
        }}
      />
      <svg className="dropzone-icon" viewBox="0 0 48 48" fill="none" stroke="currentColor"
        strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M6 14V8a2 2 0 0 1 2-2h6M34 6h6a2 2 0 0 1 2 2v6M42 34v6a2 2 0 0 1-2 2h-6M14 42H8a2 2 0 0 1-2-2v-6" />
        <circle cx="24" cy="24" r="8" />
        <path d="M24 12v4M24 32v4M12 24h4M32 24h4" />
        <circle cx="24" cy="24" r="2" fill="currentColor" />
      </svg>
      <p className="dropzone-title">Drop an image or video here</p>
      <p className="muted">or click to choose a file (up to 200 MB)</p>
    </div>
  )
}
