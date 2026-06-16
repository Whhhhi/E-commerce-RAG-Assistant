import axios from 'axios'
import type { ChatRequest, ChatResponse, UploadResponse } from '@/types'

const api = axios.create({
  baseURL: '/',
  timeout: 60000,
})

api.interceptors.response.use(
  r => r,
  error => {
    const detail = error.response?.data?.detail
    const msg = typeof detail === 'string' ? detail : detail?.message || error.message
    return Promise.reject(new Error(msg))
  }
)

export async function sendMessage(req: ChatRequest): Promise<ChatResponse> {
  const response = await api.post<ChatResponse>('/chat', req)
  return response.data
}

export async function checkHealth(): Promise<boolean> {
  try {
    const resp = await api.get('/health')
    return resp.data?.status === 'healthy'
  } catch {
    return false
  }
}

export async function uploadFile(
  file: File,
  knowledgeBase: string,
  onProgress?: (pct: number) => void
): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('knowledge_base', knowledgeBase)

  const resp = await api.post<UploadResponse>('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => {
      if (e.total && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })
  return resp.data
}
