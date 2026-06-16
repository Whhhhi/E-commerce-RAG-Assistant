<script setup lang="ts">
import { ref, nextTick } from 'vue'

const props = defineProps<{
  loading: boolean
  online: boolean
}>()

const emit = defineEmits<{
  send: [text: string]
  upload: []
}>()

const inputText = ref('')
const textareaRef = ref<HTMLTextAreaElement | null>(null)

function adjustHeight() {
  if (textareaRef.value) {
    textareaRef.value.style.height = 'auto'
    textareaRef.value.style.height = Math.min(textareaRef.value.scrollHeight, 160) + 'px'
  }
}

function handleSend() {
  const text = inputText.value.trim()
  if (!text || props.loading || !props.online) return
  emit('send', text)
  inputText.value = ''
  adjustHeight()
  nextTick(() => textareaRef.value?.focus())
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}
</script>

<template>
  <div class="input-container">
    <div class="input-bar">
      <!-- Upload button -->
      <button
        class="action-btn"
        title="上传文档"
        :disabled="loading || !online"
        @click="emit('upload')"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>

      <!-- Textarea -->
      <textarea
        ref="textareaRef"
        v-model="inputText"
        class="text-input"
        :placeholder="online ? '输入您的问题… (Enter 发送, Shift+Enter 换行)' : '服务未连接…'"
        :disabled="loading || !online"
        rows="1"
        @keydown="onKeydown"
        @input="adjustHeight"
      ></textarea>

      <!-- Send button -->
      <button
        class="send-btn"
        :class="{ active: inputText.trim() && online }"
        :disabled="!inputText.trim() || loading || !online"
        @click="handleSend"
        :title="loading ? '发送中…' : '发送'"
      >
        <svg v-if="!loading" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
          <path d="M22 2L11 13" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span v-else class="spinner"></span>
      </button>
    </div>

    <!-- Offline banner -->
    <Transition name="fade">
      <div v-if="!online" class="offline-banner">
        ⚠️ 后端服务未连接 —
        <code>uv run uvicorn app.main:app --reload --port 8000</code>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.input-container {
  flex-shrink: 0;
  background: white;
  border-top: 1px solid var(--border);
  padding: 12px 16px 16px;
}
.input-bar {
  display: flex;
  gap: 10px;
  align-items: flex-end;
}

/* ── Action Button (Upload) ── */
.action-btn {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: white;
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}
.action-btn:hover:not(:disabled) {
  background: #f3f4f6;
  color: var(--primary);
  border-color: var(--primary);
}
.action-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ── Textarea ── */
.text-input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: 12px;
  font-size: 14px;
  font-family: inherit;
  line-height: 1.5;
  resize: none;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
  max-height: 160px;
  min-height: 40px;
  background: #f9fafb;
}
.text-input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(79,70,229,0.1);
  background: white;
}
.text-input:disabled {
  background: #f3f4f6;
  cursor: not-allowed;
}
.text-input::placeholder {
  color: var(--text-light);
}

/* ── Send Button ── */
.send-btn {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: 10px;
  border: none;
  background: #e5e7eb;
  color: #9ca3af;
  cursor: not-allowed;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}
.send-btn.active {
  background: var(--primary);
  color: white;
  cursor: pointer;
}
.send-btn.active:hover {
  background: var(--primary-hover);
  transform: scale(1.05);
}
.spinner {
  display: inline-block;
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ── Offline Banner ── */
.offline-banner {
  margin-top: 10px;
  padding: 8px 12px;
  background: #fef3c7;
  border: 1px solid #fcd34d;
  border-radius: 8px;
  font-size: 12px;
  color: #92400e;
  text-align: center;
}
.offline-banner code {
  background: rgba(0,0,0,0.06);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 11px;
}

/* ── Transitions ── */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.3s;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}
</style>
