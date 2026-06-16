<script setup lang="ts">
import { ref, computed } from 'vue'
import { marked } from 'marked'
import type { ChatMessage as ChatMessageType } from '@/types'

const props = defineProps<{
  message: ChatMessageType
}>()

const showSources = ref(false)

const isUser = computed(() => props.message.role === 'user')
const isAssistant = computed(() => props.message.role === 'assistant')
const hasSources = computed(() => !!props.message.sources?.length)

function formatTime(ts: number): string {
  const d = new Date(ts)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`
}

// Markdown → HTML
const renderedContent = computed(() => {
  try {
    return marked(props.message.content, { breaks: true, gfm: true })
  } catch {
    return props.message.content
  }
})

// Limit excerpt length
function excerpt(text: string, max = 120): string {
  return text.length > max ? text.slice(0, max) + '…' : text
}
</script>

<template>
  <div class="msg-wrapper" :class="{ user: isUser, assistant: isAssistant }">
    <!-- Bot avatar -->
    <div v-if="isAssistant" class="avatar">🤖</div>
    <div v-else class="avatar user-avatar">🧑</div>

    <div class="msg-body">
      <!-- Bubble -->
      <div class="bubble" :class="{ 'is-user': isUser, 'is-bot': isAssistant }">
        <div
          v-if="isUser"
          class="bubble-text"
        >{{ message.content }}</div>
        <div
          v-else
          class="bubble-text markdown-body"
          v-html="renderedContent"
        ></div>
      </div>

      <!-- Sources -->
      <div v-if="hasSources" class="sources-wrap">
        <button class="sources-toggle" @click="showSources = !showSources">
          <svg :class="{ rotated: showSources }" class="chevron" viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
            <path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
          </svg>
          <span>{{ showSources ? '收起来源' : '查看来源' }} ({{ message.sources!.length }})</span>
        </button>
        <Transition name="fade">
          <div v-if="showSources" class="sources-list">
            <div
              v-for="(src, i) in message.sources"
              :key="i"
              class="source-card"
            >
              <div class="source-header">
                <span class="source-name">
                  <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path d="M3 3.5A1.5 1.5 0 014.5 2h6.879a1.5 1.5 0 011.06.44l4.122 4.12A1.5 1.5 0 0117 7.622V16.5a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 013 16.5v-13z"/></svg>
                  {{ src.document_name }}
                </span>
                <span class="source-score">
                  {{ (src.relevance_score * 100).toFixed(0) }}%
                </span>
              </div>
              <p class="source-excerpt">{{ excerpt(src.content_excerpt) }}</p>
            </div>
          </div>
        </Transition>
      </div>

      <!-- Timestamp -->
      <span class="time">{{ formatTime(message.timestamp) }}</span>
    </div>
  </div>
</template>

<style scoped>
.msg-wrapper {
  display: flex;
  gap: 10px;
  padding: 6px 0;
  max-width: 100%;
}
.msg-wrapper.user {
  flex-direction: row-reverse;
}

/* ── Avatar ── */
.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
  margin-top: 4px;
  background: #e5e7eb;
}
.user-avatar {
  background: #c7d2fe;
}

/* ── Body ── */
.msg-body {
  max-width: 75%;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.user .msg-body {
  align-items: flex-end;
}

/* ── Bubble ── */
.bubble {
  padding: 10px 16px;
  line-height: 1.5;
  font-size: 14px;
  word-break: break-word;
  white-space: pre-wrap;
}
.bubble.is-bot {
  background: var(--bot-bubble);
  border-radius: 0 var(--radius-lg) var(--radius-lg) var(--radius-lg);
  box-shadow: var(--shadow);
  color: var(--text);
}
.bubble.is-user {
  background: var(--user-bubble);
  border-radius: var(--radius-lg) 0 var(--radius-lg) var(--radius-lg);
  color: var(--user-text);
}
.bubble-text {
  white-space: pre-wrap;
}
.bubble-text :deep(p) {
  margin: 0;
}

/* ── Sources ── */
.sources-wrap {
  margin-top: 4px;
}
.sources-toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: none;
  border: none;
  color: var(--primary);
  font-size: 12px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 6px;
  transition: background 0.15s;
}
.sources-toggle:hover {
  background: var(--primary-light);
}
.chevron {
  transition: transform 0.2s;
}
.chevron.rotated {
  transform: rotate(180deg);
}
.sources-list {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.source-card {
  background: #f9fafb;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 13px;
}
.source-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.source-name {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--text);
  font-weight: 500;
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.source-name svg {
  flex-shrink: 0;
  color: var(--text-secondary);
}
.source-score {
  font-size: 11px;
  font-weight: 600;
  color: var(--primary);
  flex-shrink: 0;
  margin-left: 8px;
}
.source-excerpt {
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.5;
  margin: 0;
}

/* ── Time ── */
.time {
  font-size: 11px;
  color: var(--text-light);
  padding: 0 4px;
  margin-top: 2px;
}
.user .time {
  text-align: right;
}

/* ── Transitions ── */
.fade-enter-active, .fade-leave-active {
  transition: all 0.2s ease;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}

/* ── Responsive ── */
@media (max-width: 640px) {
  .msg-body {
    max-width: 88%;
  }
  .bubble {
    padding: 8px 12px;
    font-size: 14px;
  }
}
</style>
