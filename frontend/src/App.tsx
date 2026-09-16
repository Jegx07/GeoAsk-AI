import { useState } from 'react'
import { Send, Upload, Earth, Download, Activity, AlertCircle } from 'lucide-react'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'

export default function App() {
  const [query, setQuery] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)

  const acceptFiles = (nextFiles: File[]) => {
    const supported = nextFiles.filter(file => /\.(tif|tiff|png|jpe?g)$/i.test(file.name))
    setFiles(supported)
    setError(supported.length === nextFiles.length ? '' : 'Unsupported file skipped. Use GeoTIFF, TIFF, PNG, or JPEG.')
  }
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      acceptFiles(Array.from(e.target.files))
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query || files.length === 0) return
    
    setLoading(true)
    setError('')
    const formData = new FormData()
    formData.append('query', query)
    files.forEach(f => formData.append('files', f))

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Analysis request failed')
      setResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to connect to backend'
      console.error(err)
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background flex flex-col items-center py-10 px-4 dark">
      <header className="mb-10 text-center flex flex-col items-center">
        <Earth className="w-12 h-12 text-primary mb-4" />
        <h1 className="text-4xl font-bold tracking-tight">GeoAsk AI</h1>
        <p className="text-muted-foreground mt-2">Ask the Earth anything.</p>
      </header>

      <main className="w-full max-w-5xl grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Left Column: Input */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Input</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              
              {/* File Upload */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Upload Satellite Imagery</label>
                <div
                  className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center bg-muted/20 ${dragging ? 'border-primary' : ''}`}
                  onDragOver={e => { e.preventDefault(); setDragging(true) }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={e => { e.preventDefault(); setDragging(false); acceptFiles(Array.from(e.dataTransfer.files)) }}
                >
                  <Upload className="w-8 h-8 text-muted-foreground mb-4" />
                  <Input type="file" accept=".tif,.tiff,.png,.jpg,.jpeg,image/tiff,image/png,image/jpeg" multiple onChange={handleFileChange} className="hidden" id="file-upload" />
                  <label htmlFor="file-upload" className="cursor-pointer text-sm text-primary hover:underline">
                    Browse files
                  </label>
                  <p className="text-xs text-muted-foreground mt-2">
                    {files.length > 0 ? files.map(file => file.name).join(', ') : "Drop GeoTIFF, PNG, or JPEG files here"}
                  </p>
                </div>
              </div>

              {error && <p className="flex items-center gap-2 text-sm text-destructive"><AlertCircle className="h-4 w-4" />{error}</p>}

              {/* Query Input */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Natural Language Query</label>
                <div className="flex space-x-2">
                  <Input 
                    value={query} 
                    onChange={e => setQuery(e.target.value)} 
                    placeholder="e.g., What changed between these images?" 
                    disabled={loading}
                  />
                  <Button type="submit" disabled={loading || files.length === 0 || !query}>
                    {loading ? <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full" /> : <Send className="w-4 h-4" />}
                  </Button>
                </div>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Right Column: Output */}
        <Card className="flex flex-col h-full min-h-[500px]">
          <CardHeader>
            <CardTitle>Results</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-auto">
            {!result && !loading && (
              <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                Awaiting input...
              </div>
            )}
            {loading && (
              <div className="h-full flex flex-col items-center justify-center space-y-4">
                <Earth className="w-10 h-10 animate-pulse text-primary" />
                <p className="text-sm text-muted-foreground animate-pulse">Analysing imagery...</p>
              </div>
            )}
            {result && !loading && (
              <div className="space-y-6">
                
                {/* Answer */}
                <div>
                  <h4 className="font-medium text-primary mb-2 flex items-center gap-2">
                    Answer 
                    <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded-full">
                      Confidence: {Math.round(result.confidence.overall * 100)}%
                    </span>
                  </h4>
                  <div className="bg-muted/30 p-4 rounded-md text-sm whitespace-pre-wrap leading-relaxed">
                    {result.answer}
                  </div>
                </div>

                {/* Evidence Visuals */}
                {result.visual_outputs?.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-2 text-sm">Visual Evidence</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {result.visual_outputs.map((path: string, i: number) => {
                        const filename = path.split('/').pop() || path.split('\\').pop()
                        return (
                          <div key={i} className="border rounded-md overflow-hidden">
                            <img src={`/api/static/uploads/results/${filename}`} alt="Evidence" className="w-full object-cover" />
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                {/* Download Report */}
                {result.report_path && (
                  <Button variant="outline" className="w-full mt-4" asChild>
                    <a href={`/api/sessions/${result.session_id}/report`} download>
                      <Download className="w-4 h-4 mr-2" /> Download Full Report
                    </a>
                  </Button>
                )}

                {result.execution_trace?.events?.length > 0 && (
                  <details className="border rounded-md p-3">
                    <summary className="cursor-pointer flex items-center gap-2 text-sm font-medium">
                      <Activity className="h-4 w-4" /> Execution trace ({result.execution_trace.events.length} events)
                    </summary>
                    <ol className="mt-3 space-y-2 border-l pl-4 text-xs text-muted-foreground">
                      {result.execution_trace.events.map((event: { event_type: string; message: string; timestamp: string }, i: number) => (
                        <li key={`${event.timestamp}-${i}`}>
                          <span className="font-medium text-foreground">{event.event_type}</span>: {event.message}
                        </li>
                      ))}
                    </ol>
                  </details>
                )}
                
              </div>
            )}
          </CardContent>
        </Card>

      </main>
    </div>
  )
}
