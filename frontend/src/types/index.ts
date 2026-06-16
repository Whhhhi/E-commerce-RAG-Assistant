export interface Source {
  document_id: string
  document_name: string
  chunk_index: number
  relevance_score: number
  content_excerpt: string
}

export interface ChatMessage {
  id: string
  content: string
  role: 'user' | 'assistant'
  sources?: Source[]
  timestamp: number
}

export interface ChatRequest {
  message: string
  session_id: string
  user_id?: string
}

export interface ChatResponse {
  answer: string
  sources: Source[]
  session_id: string
  intent: string
  conversation_id: string
  latency_ms?: number
}

export interface UploadResponse {
  document_id: string
  file_name: string
  knowledge_base: string
  chunks: number
  status: string
  elapsed_seconds: number
  indexed_chunks?: number
  failed_chunks?: number
}

export interface QuickAction {
  label: string
  text: string
  icon: string
}
