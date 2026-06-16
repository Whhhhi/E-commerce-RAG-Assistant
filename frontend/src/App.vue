<script setup lang="ts">
import { ref, onMounted, nextTick, watch } from 'vue'
import ChatMessage from '@/components/ChatMessage.vue'
import ChatInput from '@/components/ChatInput.vue'
import FileUpload from '@/components/FileUpload.vue'
import { sendMessage, checkHealth } from '@/api/chat'
import type { ChatMessage as ChatMessageType, QuickAction } from '@/types'

// ── State ──
const messages = ref<ChatMessageType[]>([])
const loading = ref(false)
const messagesContainer = ref<HTMLElement | null>(null)
const sessionId = ref('')
const online = ref(false)
const checkingHealth = ref(true)
const showUpload = ref(false)

// ── Quick actions ──
const quickActions: QuickAction[] = [
  { label: '📦 查订单', text: '我想查一下我的订单状态', icon: '📦' },
  { label: '🔄 退换货', text: '请问退换货政策是什么？', icon: '🔄' },
  { label: '🏷️ 商品咨询', text: '有什么热销商品推荐？', icon: '🏷️' },
  { label: '📞 人工服务', text: '我想转人工客服', icon: '📞' },
]
const showQuickActions = ref(true)

// ── Helpers ──
function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2, 6)
}

function genSessionId(): string {
  let id = localStorage.getItem('chat_session_id')
  if (!id) {
    id = 'sess_' + genId()
    localStorage.setItem('chat_session_id', id)
  }
  return id
}

function scrollBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTo({
        top: messagesContainer.value.scrollHeight,
        behavior: 'smooth',
      })
    }
  })
}

// ── Health check ──
async function ping() {
  checkingHealth.value = true
  online.value = await checkHealth()
  checkingHealth.value = false
}

// ── Send message ──
async function handleSend(text: string) {
  showQuickActions.value = false

  const userMsg: ChatMessageType = {
    id: genId(),
    content: text,
    role: 'user',
    timestamp: Date.now(),
  }
  messages.value.push(userMsg)
  scrollBottom()

  loading.value = true
  try {
    const resp = await sendMessage({
      message: text,
      session_id: sessionId.value,
      user_id: 'user_' + genId().substr(0, 6),
    })
    const botMsg: ChatMessageType = {
      id: genId(),
      content: resp.answer,
      role: 'assistant',
      sources: resp.sources,
      timestamp: Date.now(),
    }
    messages.value.push(botMsg)
  } catch (e: any) {
    messages.value.push({
      id: genId(),
      content: `**服务暂时不可用**\n\n${e.message || '请检查后端服务是否已启动，或稍后重试。'}`,
      role: 'assistant',
      timestamp: Date.now(),
    })
  } finally {
    loading.value = false
    scrollBottom()
  }
}

// ── Quick action click ──
function onQuickAction(action: QuickAction) {
  handleSend(action.text)
}

// ── File uploaded ──
function onFileUploaded(msg: string) {
  showUpload.value = false
  messages.value.push({
    id: genId(),
    content: `✅ **文档上传成功**\n\n${msg}`,
    role: 'assistant',
    timestamp: Date.now(),
  })
  scrollBottom()
}

// ── Init ──
onMounted(async () => {
  sessionId.value = genSessionId()
  await ping()
  // 定时检测
  setInterval(ping, 30000)

  messages.value.push({
    id: genId(),
    content: online.value
      ? '您好！我是 **AI 智能客服**，可以帮您查询订单、了解退换货政策、推荐商品等。请选择下方快捷问题或直接输入。'
      : '⚠️ **后端服务未连接**\n\n请确保 FastAPI 服务已启动（`uv run uvicorn app.main:app --reload`），然后刷新页面。',
    role: 'assistant',
    timestamp: Date.now(),
  })
})

// 连接恢复后提示
watch(online, (now, prev) => {
  if (now && prev === false) {
    messages.value.push({
      id: genId(),
      content: '✅ **服务已恢复连接**，您可以继续提问了。',
      role: 'assistant',
      timestamp: Date.now(),
    })
    scrollBottom()
  }
})
</script>

<template>
  <div class="chat-app">
    <!-- ── Header ── -->
    <header class="chat-header">
      <div class="header-left">
        <div class="brand">
          <span class="brand-icon">🛒</span>
          <div class="brand-text">
            <span class="brand-name">智能客服</span>
            <span class="brand-sub">AI 驱动的电商助手</span>
          </div>
        </div>
      </div>
      <div class="header-right">
        <button class="header-btn" title="上传文档" @click="showUpload = !showUpload">
          <span>📄</span>
        </button>
        <div class="status-badge" :class="{ online, offline: !online }">
          <span class="status-dot"></span>
          <span class="status-text">{{ checkingHealth ? '检测中' : online ? '服务在线' : '未连接' }}</span>
        </div>
      </div>
    </header>

    <!-- ── Upload Panel ── -->
    <Transition name="slide">
      <div v-if="showUpload" class="upload-panel">
        <FileUpload @uploaded="onFileUploaded" @close="showUpload = false" />
      </div>
    </Transition>

    <!-- ── Messages ── -->
    <main ref="messagesContainer" class="messages-container">
      <div v-if="messages.length === 0" class="empty-state">
        <div class="empty-icon">💬</div>
        <p>开始您的对话吧</p>
      </div>

      <ChatMessage
        v-for="msg in messages"
        :key="msg.id"
        :message="msg"
      />

      <!-- Loading -->
      <div v-if="loading" class="loading-row">
        <div class="loading-avatar">🤖</div>
        <div class="loading-bubble">
          <span class="typing-dot" style="animation-delay:0s"></span>
          <span class="typing-dot" style="animation-delay:0.16s"></span>
          <span class="typing-dot" style="animation-delay:0.32s"></span>
        </div>
      </div>

      <!-- Quick actions -->
      <div v-if="showQuickActions && !loading" class="quick-actions">
        <p class="quick-actions-title">快捷问题</p>
        <div class="quick-actions-grid">
          <button
            v-for="(action, i) in quickActions"
            :key="i"
            class="quick-action-btn"
            @click="onQuickAction(action)"
          >
            <span class="qa-icon">{{ action.icon }}</span>
            <span class="qa-label">{{ action.label }}</span>
          </button>
        </div>
      </div>
    </main>

    <!-- ── Input ── -->
    <ChatInput
      :loading="loading"
      :online="online"
      @send="handleSend"
      @upload="showUpload = !showUpload"
    />
  </div>
</template>

<style scoped>
.chat-app {
  width: 100%;
  max-width: 800px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--bg-chat);
  box-shadow: var(--shadow-lg);
  position: relative;
  overflow: hidden;
}

/* ── Header ── */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  color: white;
  flex-shrink: 0;
  z-index: 10;
}
.header-left {
  display: flex;
  align-items: center;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
}
.brand-icon {
  font-size: 28px;
  line-height: 1;
}
.brand-text {
  display: flex;
  flex-direction: column;
}
.brand-name {
  font-size: 17px;
  font-weight: 700;
  letter-spacing: 0.5px;
}
.brand-sub {
  font-size: 11px;
  opacity: 0.8;
  margin-top: 1px;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}
.header-btn {
  background: rgba(255,255,255,0.15);
  border: none;
  color: white;
  width: 34px;
  height: 34px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.2s;
}
.header-btn:hover {
  background: rgba(255,255,255,0.25);
}
.status-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: var(--radius-full);
  background: rgba(255,255,255,0.15);
}
.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #9ca3af;
  transition: background 0.3s;
}
.status-badge.online .status-dot {
  background: #34d399;
  box-shadow: 0 0 6px rgba(52,211,153,0.6);
}
.status-badge.offline .status-dot {
  background: #f87171;
}

/* ── Upload Panel ── */
.upload-panel {
  flex-shrink: 0;
  border-bottom: 1px solid var(--border);
  background: #f9fafb;
}
.slide-enter-active, .slide-leave-active {
  transition: all 0.25s ease;
}
.slide-enter-from, .slide-leave-to {
  max-height: 0;
  opacity: 0;
  overflow: hidden;
}
.slide-enter-to, .slide-leave-from {
  max-height: 300px;
  opacity: 1;
}

/* ── Messages ── */
.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: var(--bg);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-light);
  gap: 8px;
}
.empty-icon {
  font-size: 48px;
  opacity: 0.5;
}

/* ── Loading ── */
.loading-row {
  display: flex;
  gap: 10px;
  padding: 8px 0;
  align-items: flex-end;
}
.loading-avatar {
  font-size: 28px;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.loading-bubble {
  background: var(--bot-bubble);
  padding: 14px 18px;
  border-radius: 0 var(--radius-lg) var(--radius-lg) var(--radius-lg);
  box-shadow: var(--shadow-sm);
  display: flex;
  gap: 5px;
  align-items: center;
}
.typing-dot {
  width: 8px;
  height: 8px;
  background: var(--primary);
  border-radius: 50%;
  animation: typingBounce 1.4s infinite ease-in-out both;
}
@keyframes typingBounce {
  0%, 80%, 100% { transform: scale(0.4); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}

/* ── Quick Actions ── */
.quick-actions {
  padding: 16px 4px 8px;
}
.quick-actions-title {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 10px;
  font-weight: 500;
}
.quick-actions-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
.quick-action-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 12px 8px;
  background: white;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: all 0.2s;
  font-size: 13px;
  color: var(--text);
}
.quick-action-btn:hover {
  border-color: var(--primary);
  background: var(--primary-light);
  transform: translateY(-1px);
  box-shadow: var(--shadow);
}
.qa-icon {
  font-size: 20px;
}
.qa-label {
  font-weight: 500;
  white-space: nowrap;
}

/* ── Responsive ── */
@media (max-width: 640px) {
  .chat-app {
    max-width: 100%;
    height: 100dvh;
  }
  .messages-container {
    padding: 12px;
  }
  .quick-actions-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  .brand-sub {
    display: none;
  }
}
</style>
