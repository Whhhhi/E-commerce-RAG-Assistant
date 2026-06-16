<script setup lang="ts">
import { ref } from 'vue'
import { uploadFile } from '@/api/chat'

const emit = defineEmits<{
  uploaded: [msg: string]
  close: []
}>()

const file = ref<File | null>(null)
const dragOver = ref(false)
const uploading = ref(false)
const progress = ref(0)
const kb = ref('product')
const error = ref('')
const success = ref('')

const KB_OPTIONS = [
  { value: 'product', label: '🛍️ 商品知识库' },
  { value: 'policy', label: '📋 政策知识库' },
]

function onFileSelected(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files?.length) {
    validateFile(input.files[0])
  }
}

function onDrop(e: DragEvent) {
  dragOver.value = false
  if (e.dataTransfer?.files.length) {
    validateFile(e.dataTransfer.files[0])
  }
}

function validateFile(f: File) {
  error.value = ''
  success.value = ''

  const ext = f.name.split('.').pop()?.toLowerCase()
  const allowed = ['txt', 'pdf', 'md', 'docx', 'csv', 'json', 'html']
  if (!ext || !allowed.includes(ext)) {
    error.value = `不支持 .${ext} 格式，支持: ${allowed.join(', ')}`
    return
  }
  if (f.size > 20 * 1024 * 1024) {
    error.value = '文件超过 20MB 限制'
    return
  }
  file.value = f
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / 1024 / 1024).toFixed(1) + ' MB'
}

async function doUpload() {
  if (!file.value) return
  uploading.value = true
  progress.value = 0
  error.value = ''
  success.value = ''

  try {
    const resp = await uploadFile(file.value, kb.value, (pct) => {
      progress.value = pct
    })
    success.value = `✅ 上传成功！${resp.chunks} 个文档块已存入 ${resp.knowledge_base} 知识库`
    emit('uploaded', `${resp.chunks} 个文档块已导入「${resp.knowledge_base}」知识库`)
  } catch (e: any) {
    error.value = e.message || '上传失败，请重试'
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <div class="upload-panel">
    <div class="upload-header">
      <h3>📄 上传文档</h3>
      <button class="close-btn" @click="emit('close')">✕</button>
    </div>

    <!-- Knowledge base selector -->
    <div class="kb-selector">
      <label class="kb-label">导入到：</label>
      <div class="kb-options">
        <button
          v-for="opt in KB_OPTIONS"
          :key="opt.value"
          class="kb-btn"
          :class="{ active: kb === opt.value }"
          :disabled="uploading"
          @click="kb = opt.value"
        >
          {{ opt.label }}
        </button>
      </div>
    </div>

    <!-- Drop zone -->
    <div
      class="drop-zone"
      :class="{ dragover: dragOver, hasFile: !!file }"
      @dragover.prevent="dragOver = true"
      @dragleave="dragOver = false"
      @drop.prevent="onDrop"
      @click="!uploading && ($refs.fileInput as HTMLElement)?.click()"
    >
      <template v-if="!file">
        <div class="drop-icon">📁</div>
        <p class="drop-text">拖放文件到此处，或点击选择</p>
        <p class="drop-hint">支持 TXT / PDF / DOCX / MD / CSV</p>
      </template>
      <template v-else>
        <div class="file-info">
          <span class="file-icon">📄</span>
          <div class="file-detail">
            <span class="file-name">{{ file.name }}</span>
            <span class="file-size">{{ formatSize(file.size) }}</span>
          </div>
          <button
            v-if="!uploading"
            class="file-remove"
            @click.stop="file = null; error=''; success=''"
          >✕</button>
        </div>
      </template>
      <input
        ref="fileInput"
        type="file"
        accept=".txt,.pdf,.md,.docx,.csv,.json,.html"
        hidden
        @change="onFileSelected"
      />
    </div>

    <!-- Progress bar -->
    <div v-if="uploading" class="progress-wrap">
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progress + '%' }"></div>
      </div>
      <span class="progress-text">{{ progress }}%</span>
    </div>

    <!-- Error -->
    <div v-if="error" class="msg msg-error">{{ error }}</div>

    <!-- Success -->
    <div v-if="success" class="msg msg-success">{{ success }}</div>

    <!-- Upload button -->
    <button
      v-if="file && !uploading"
      class="upload-btn"
      @click="doUpload"
    >
      🚀 上传到 {{ kb === 'product' ? '商品' : '政策' }}知识库
    </button>
  </div>
</template>

<style scoped>
.upload-panel {
  padding: 16px 20px;
}
.upload-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}
.upload-header h3 {
  font-size: 15px;
  font-weight: 600;
}
.close-btn {
  background: none;
  border: none;
  font-size: 18px;
  color: var(--text-secondary);
  cursor: pointer;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.close-btn:hover {
  background: #f3f4f6;
}

/* ── KB Selector ── */
.kb-selector {
  margin-bottom: 12px;
}
.kb-label {
  font-size: 13px;
  color: var(--text-secondary);
  display: block;
  margin-bottom: 6px;
}
.kb-options {
  display: flex;
  gap: 8px;
}
.kb-btn {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: white;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}
.kb-btn.active {
  border-color: var(--primary);
  background: var(--primary-light);
  color: var(--primary);
  font-weight: 500;
}
.kb-btn:hover:not(:disabled):not(.active) {
  border-color: var(--primary);
}
.kb-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* ── Drop Zone ── */
.drop-zone {
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  padding: 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: #fafafa;
}
.drop-zone:hover {
  border-color: var(--primary);
  background: var(--primary-light);
}
.drop-zone.dragover {
  border-color: var(--primary);
  background: var(--primary-light);
  transform: scale(1.01);
}
.drop-zone.hasFile {
  border-style: solid;
  border-color: var(--primary);
  background: #f5f3ff;
  cursor: default;
}
.drop-icon {
  font-size: 36px;
  margin-bottom: 8px;
}
.drop-text {
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 4px;
}
.drop-hint {
  font-size: 12px;
  color: var(--text-light);
}

/* ── File Info ── */
.file-info {
  display: flex;
  align-items: center;
  gap: 10px;
}
.file-icon {
  font-size: 28px;
}
.file-detail {
  flex: 1;
  text-align: left;
}
.file-name {
  font-size: 14px;
  font-weight: 500;
  display: block;
  word-break: break-all;
}
.file-size {
  font-size: 12px;
  color: var(--text-secondary);
}
.file-remove {
  background: none;
  border: none;
  color: var(--text-light);
  cursor: pointer;
  font-size: 16px;
  padding: 4px;
  border-radius: 4px;
}
.file-remove:hover {
  background: rgba(0,0,0,0.05);
  color: var(--red);
}

/* ── Progress ── */
.progress-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
}
.progress-bar {
  flex: 1;
  height: 6px;
  background: #e5e7eb;
  border-radius: 3px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--primary);
  border-radius: 3px;
  transition: width 0.3s;
}
.progress-text {
  font-size: 12px;
  color: var(--text-secondary);
  min-width: 36px;
}

/* ── Messages ── */
.msg {
  margin-top: 10px;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 13px;
}
.msg-error {
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #b91c1c;
}
.msg-success {
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  color: #15803d;
}

/* ── Upload Button ── */
.upload-btn {
  width: 100%;
  margin-top: 12px;
  padding: 10px;
  background: var(--primary);
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}
.upload-btn:hover {
  background: var(--primary-hover);
}
</style>
