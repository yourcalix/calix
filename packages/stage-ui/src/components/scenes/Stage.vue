<script setup lang="ts">
import type { DuckDBWasmDrizzleDatabase } from '@proj-airi/drizzle-duckdb-wasm'
import type { Live2DLipSync, Live2DLipSyncOptions } from '@proj-airi/model-driver-lipsync'
import type { Profile } from '@proj-airi/model-driver-lipsync/shared/wlipsync'
import type { SpeechProviderWithExtraOptions } from '@xsai-ext/providers/utils'
import type { UnElevenLabsOptions } from 'unspeech'

import type { Emotion } from '../../constants/emotions'

import embedWorkerURL from '@xsai-transformers/embed/worker?worker&url'

import { DBStorageType, drizzle, DuckDBAccessMode } from '@proj-airi/drizzle-duckdb-wasm'
import { getImportUrlBundles } from '@proj-airi/drizzle-duckdb-wasm/bundles/import-url-browser'
import { createLive2DLipSync } from '@proj-airi/model-driver-lipsync'
import { wlipsyncProfile } from '@proj-airi/model-driver-lipsync/shared/wlipsync'
import { createPlaybackManager, createSpeechPipeline } from '@proj-airi/pipelines-audio'
import { ContextUpdateStrategy } from '@proj-airi/server-sdk'
import { Live2DScene, useLive2d } from '@proj-airi/stage-ui-live2d'
import { ThreeScene, useModelStore } from '@proj-airi/stage-ui-three'
import { animations } from '@proj-airi/stage-ui-three/assets/vrm'
import { createQueue } from '@proj-airi/stream-kit'
import { useBroadcastChannel } from '@vueuse/core'
import { createEmbedProvider } from '@xsai-transformers/embed'
import { embed } from '@xsai/embed'
import { generateSpeech } from '@xsai/generate-speech'
import { storeToRefs } from 'pinia'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'

import { useDelayMessageQueue, useEmotionsMessageQueue } from '../../composables/queues'
import { llmInferenceEndToken } from '../../constants'
import { EMOTION_EmotionMotionName_value, EMOTION_VRMExpressionName_value, EmotionThinkMotionName } from '../../constants/emotions'
import { useAudioContext, useSpeakingStore } from '../../stores/audio'
import { useChatOrchestratorStore } from '../../stores/chat'
import { useChatContextStore } from '../../stores/chat/context-store'
import { useAiriCardStore } from '../../stores/modules'
import { useSpeechStore } from '../../stores/modules/speech'
import { useProvidersStore } from '../../stores/providers'
import { useSettings } from '../../stores/settings'
import { useSpeechRuntimeStore } from '../../stores/speech-runtime'

withDefaults(defineProps<{
  paused?: boolean
  focusAt: { x: number, y: number }
  xOffset?: number | string
  yOffset?: number | string
  scale?: number
}>(), { paused: false, scale: 1 })

const componentState = defineModel<'pending' | 'loading' | 'mounted'>('state', { default: 'pending' })

const db = ref<DuckDBWasmDrizzleDatabase>()
let embedProvider: ReturnType<typeof createEmbedProvider> | null = null
const chatContextStore = useChatContextStore()

const vrmViewerRef = ref<InstanceType<typeof ThreeScene>>()
const live2dSceneRef = ref<InstanceType<typeof Live2DScene>>()

const settingsStore = useSettings()
const {
  stageModelRenderer,
  stageViewControlsEnabled,
  live2dDisableFocus,
  stageModelSelectedUrl,
  stageModelSelected,
  themeColorsHue,
  themeColorsHueDynamic,
  live2dIdleAnimationEnabled,
  live2dAutoBlinkEnabled,
  live2dForceAutoBlinkEnabled,
  live2dShadowEnabled,
  memoryEnabled,
} = storeToRefs(settingsStore)
const { mouthOpenSize } = storeToRefs(useSpeakingStore())
const { audioContext } = useAudioContext()
const currentAudioSource = ref<AudioBufferSourceNode>()

const { onBeforeMessageComposed, onBeforeSend, onTokenLiteral, onTokenSpecial, onStreamEnd, onAssistantResponseEnd } = useChatOrchestratorStore()
const chatHookCleanups: Array<() => void> = []
// WORKAROUND: clear previous handlers on unmount to avoid duplicate calls when this component remounts.
//             We keep per-hook disposers instead of wiping the global chat hooks to play nicely with
//             cross-window broadcast wiring.

const providersStore = useProvidersStore()
const live2dStore = useLive2d()
const vrmStore = useModelStore()

const showStage = ref(true)
const viewUpdateCleanups: Array<() => void> = []

// Caption + Presentation broadcast channels
type CaptionChannelEvent
  = | { type: 'caption-speaker', text: string }
    | { type: 'caption-assistant', text: string }
const { post: postCaption } = useBroadcastChannel<CaptionChannelEvent, CaptionChannelEvent>({ name: 'airi-caption-overlay' })
const assistantCaption = ref('')

type PresentEvent
  = | { type: 'assistant-reset' }
    | { type: 'assistant-append', text: string }
const { post: postPresent } = useBroadcastChannel<PresentEvent, PresentEvent>({ name: 'airi-chat-present' })

viewUpdateCleanups.push(live2dStore.onShouldUpdateView(async () => {
  showStage.value = false
  await settingsStore.updateStageModel()
  setTimeout(() => {
    showStage.value = true
  }, 100)
}))

viewUpdateCleanups.push(vrmStore.onShouldUpdateView(async () => {
  showStage.value = false
  await settingsStore.updateStageModel()
  setTimeout(() => {
    showStage.value = true
  }, 100)
}))

const audioAnalyser = ref<AnalyserNode>()
const nowSpeaking = ref(false)
const lipSyncStarted = ref(false)
const lipSyncLoopId = ref<number>()
const live2dLipSync = ref<Live2DLipSync>()
const live2dLipSyncOptions: Live2DLipSyncOptions = { mouthUpdateIntervalMs: 50, mouthLerpWindowMs: 50 }

const { activeCard } = storeToRefs(useAiriCardStore())
const speechStore = useSpeechStore()
const { ssmlEnabled, activeSpeechProvider, activeSpeechModel, activeSpeechVoice, pitch } = storeToRefs(speechStore)
const activeCardId = computed(() => activeCard.value?.name ?? 'default')
const speechRuntimeStore = useSpeechRuntimeStore()

const { currentMotion } = storeToRefs(useLive2d())

const memoryEmbeddingModel = 'Xenova/nomic-embed-text-v1'
const memorySource = { plugin: 'memory', instanceId: 'local' }
const memoryContextId = 'memory'
let memorySchemaReady: Promise<void> | undefined
let canEmbedForMemory = true

function normalizeMemoryText(text: string) {
  return text.replaceAll(/\s+/g, ' ').trim()
}

function ensureEmbedProvider() {
  if (embedProvider)
    return embedProvider
  if (typeof window === 'undefined' || typeof Worker === 'undefined')
    return null
  embedProvider = createEmbedProvider({ worker: new Worker(embedWorkerURL, { type: 'module' }) })
  return embedProvider
}

async function initMemoryDb() {
  if (db.value)
    return
  db.value = drizzle({
    connection: {
      bundles: getImportUrlBundles(),
      storage: { type: DBStorageType.ORIGIN_PRIVATE_FS, path: 'airi-memory.duckdb', accessMode: DuckDBAccessMode.READ_WRITE },
    },
  })
  await ensureMemorySchema()
}

async function ensureMemorySchema() {
  if (!memoryEnabled.value)
    return
  if (!db.value)
    return
  if (!memorySchemaReady) {
    memorySchemaReady = (async () => {
      const client = await db.value!.$client
      await client.query(`
        CREATE TABLE IF NOT EXISTS chat_memory (
          id VARCHAR,
          cardId VARCHAR,
          role VARCHAR,
          text VARCHAR,
          createdAt BIGINT,
          embedding FLOAT[]
        );
      `)
      await client.query(`CREATE INDEX IF NOT EXISTS chat_memory_card_time ON chat_memory(cardId, createdAt);`)
    })()
  }
  await memorySchemaReady
}

async function tryEmbedText(text: string): Promise<number[] | null> {
  if (!canEmbedForMemory)
    return null
  if (typeof window === 'undefined')
    return null
  const provider = ensureEmbedProvider()
  if (!provider)
    return null

  try {
    const res = await embed({
      ...provider.embed(memoryEmbeddingModel),
      input: text,
    })
    const embedding = (res as any)?.embedding as unknown
    if (Array.isArray(embedding))
      return embedding as number[]
    if (embedding && typeof (embedding as any).length === 'number')
      return Array.from(embedding as any)
    return null
  }
  catch {
    canEmbedForMemory = false
    return null
  }
}

async function persistMemoryEntry(role: 'user' | 'assistant', text: string, createdAt: number) {
  if (!memoryEnabled.value)
    return
  const normalized = normalizeMemoryText(text)
  if (!normalized)
    return
  await initMemoryDb()

  await ensureMemorySchema()
  const embedding = await tryEmbedText(normalized)
  if (!db.value)
    return
  const client = await db.value.$client
  const id = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`
  await client.query(
    `INSERT INTO chat_memory (id, cardId, role, text, createdAt, embedding) VALUES (?, ?, ?, ?, ?, ?);`,
    [id, activeCardId.value, role, normalized, createdAt, embedding],
  )
}

function tokenizeForMemory(text: string) {
  return normalizeMemoryText(text)
    .toLowerCase()
    .split(/[^a-z0-9\u4E00-\u9FFF]+/g)
    .filter(Boolean)
}

function overlapScore(a: string, b: string) {
  const ta = new Set(tokenizeForMemory(a))
  const tb = new Set(tokenizeForMemory(b))
  if (ta.size === 0 || tb.size === 0)
    return 0
  let hit = 0
  for (const t of ta) {
    if (tb.has(t))
      hit += 1
  }
  return hit / Math.max(ta.size, 1)
}

async function updateMemoryContextForPrompt(prompt: string) {
  if (!memoryEnabled.value)
    return
  await initMemoryDb()
  if (!db.value)
    return

  const normalizedPrompt = normalizeMemoryText(prompt)
  if (!normalizedPrompt)
    return

  await ensureMemorySchema()
  const client = await db.value.$client
  const rows = await client.query(
    `SELECT id, role, text, createdAt FROM chat_memory WHERE cardId = ? ORDER BY createdAt DESC LIMIT 120;`,
    [activeCardId.value],
  ) as Array<{ id: string, role: string, text: string, createdAt: bigint | number }>

  const recents = [...rows].reverse().slice(-10)
  const scored = rows
    .map((r) => {
      const createdAt = typeof r.createdAt === 'bigint' ? Number(r.createdAt) : r.createdAt
      return { ...r, createdAt, score: overlapScore(normalizedPrompt, r.text) }
    })
    .filter(r => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 6)

  const pickedIds = new Set<string>()
  const picked: Array<{ role: string, text: string, createdAt: number }> = []
  for (const r of [...scored].reverse()) {
    if (pickedIds.has(r.id))
      continue
    pickedIds.add(r.id)
    picked.push({ role: r.role, text: r.text, createdAt: r.createdAt })
  }
  for (const r of recents) {
    if (pickedIds.has(r.id))
      continue
    const createdAt = typeof r.createdAt === 'bigint' ? Number(r.createdAt) : r.createdAt
    pickedIds.add(r.id)
    picked.push({ role: r.role, text: r.text, createdAt })
  }

  if (picked.length === 0)
    return

  const memoryText = picked
    .slice(-12)
    .map((m) => {
      const t = new Date(m.createdAt).toISOString()
      return `${t} ${m.role}: ${m.text}`
    })
    .join('\n')

  chatContextStore.ingestContextMessage({
    id: globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`,
    contextId: memoryContextId,
    strategy: ContextUpdateStrategy.ReplaceSelf,
    text: memoryText,
    content: memoryText,
    metadata: { source: memorySource },
    createdAt: Date.now(),
  })
}

const emotionsQueue = createQueue<Emotion>({
  handlers: [
    async (ctx) => {
      if (stageModelRenderer.value === 'vrm') {
        // console.debug("VRM emotion anime: ", ctx.data)
        const value = EMOTION_VRMExpressionName_value[ctx.data]
        if (!value)
          return

        await vrmViewerRef.value!.setExpression(value)
      }
      else if (stageModelRenderer.value === 'live2d') {
        currentMotion.value = { group: EMOTION_EmotionMotionName_value[ctx.data] }
      }
    },
  ],
})

const emotionMessageContentQueue = useEmotionsMessageQueue(emotionsQueue)
emotionMessageContentQueue.onHandlerEvent('emotion', (emotion) => {
  // eslint-disable-next-line no-console
  console.debug('emotion detected', emotion)
})

const delaysQueue = useDelayMessageQueue()
delaysQueue.onHandlerEvent('delay', (delay) => {
  // eslint-disable-next-line no-console
  console.debug('delay detected', delay)
})

// Play special token: delay or emotion
function playSpecialToken(special: string) {
  delaysQueue.enqueue(special)
  emotionMessageContentQueue.enqueue(special)
}
const lipSyncNode = ref<AudioNode>()

async function playFunction(item: Parameters<Parameters<typeof createPlaybackManager<AudioBuffer>>[0]['play']>[0], signal: AbortSignal): Promise<void> {
  return new Promise<void>(async (resolve) => {
    if (!audioContext) {
      resolve()
      return
    }

    if (!item.audio) {
      resolve()
      return
    }

    // Ensure audio context is resumed (browsers suspend it by default until user interaction)
    if (audioContext.state === 'suspended') {
      try {
        await audioContext.resume()
      }
      catch {
        resolve()
        return
      }
    }

    const source = audioContext.createBufferSource()
    currentAudioSource.value = source
    source.buffer = item.audio

    source.connect(audioContext.destination)
    if (audioAnalyser.value)
      source.connect(audioAnalyser.value)
    if (lipSyncNode.value)
      source.connect(lipSyncNode.value)

    const stopPlayback = () => {
      try {
        source.stop()
        source.disconnect()
      }
      catch {}
      if (currentAudioSource.value === source)
        currentAudioSource.value = undefined
      resolve()
    }

    if (signal.aborted) {
      stopPlayback()
      return
    }

    signal.addEventListener('abort', stopPlayback, { once: true })
    source.onended = () => {
      signal.removeEventListener('abort', stopPlayback)
      stopPlayback()
    }

    try {
      source.start(0)
    }
    catch {
      stopPlayback()
    }
  })
}

const playbackManager = createPlaybackManager<AudioBuffer>({
  play: playFunction,
  maxVoices: 1,
  maxVoicesPerOwner: 1,
  overflowPolicy: 'queue',
  ownerOverflowPolicy: 'steal-oldest',
})

const speechPipeline = createSpeechPipeline<AudioBuffer>({
  tts: async (request, signal) => {
    if (signal.aborted)
      return null

    if (!activeSpeechProvider.value)
      return null

    const provider = await providersStore.getProviderInstance(activeSpeechProvider.value) as SpeechProviderWithExtraOptions<string, UnElevenLabsOptions>
    if (!provider) {
      console.error('Failed to initialize speech provider')
      return null
    }

    if (!request.text && !request.special)
      return null

    const providerConfig = providersStore.getProviderConfig(activeSpeechProvider.value)

    // For OpenAI Compatible providers, always use provider config for model and voice
    // since these are manually configured in provider settings
    let model = activeSpeechModel.value
    let voice = activeSpeechVoice.value

    if (activeSpeechProvider.value === 'openai-compatible-audio-speech') {
      // Always prefer provider config for OpenAI Compatible (user configured it there)
      if (providerConfig?.model) {
        model = providerConfig.model as string
      }
      else {
        // Fallback to default if not in provider config
        model = 'tts-1'
        console.warn('[Speech Pipeline] OpenAI Compatible: No model in provider config, using default', { providerConfig })
      }

      if (providerConfig?.voice) {
        voice = {
          id: providerConfig.voice as string,
          name: providerConfig.voice as string,
          description: providerConfig.voice as string,
          previewURL: '',
          languages: [{ code: 'en', title: 'English' }],
          provider: activeSpeechProvider.value,
          gender: 'neutral',
        }
      }
      else {
        // Fallback to default if not in provider config
        voice = {
          id: 'alloy',
          name: 'alloy',
          description: 'alloy',
          previewURL: '',
          languages: [{ code: 'en', title: 'English' }],
          provider: activeSpeechProvider.value,
          gender: 'neutral',
        }
        console.warn('[Speech Pipeline] OpenAI Compatible: No voice in provider config, using default', { providerConfig })
      }
    }

    if (!model || !voice)
      return null

    const input = ssmlEnabled.value
      ? speechStore.generateSSML(request.text, voice, { ...providerConfig, pitch: pitch.value })
      : request.text

    try {
      const res = await generateSpeech({
        ...provider.speech(model, providerConfig),
        input,
        voice: voice.id,
      })

      if (signal.aborted || !res || res.byteLength === 0)
        return null

      const audioBuffer = await audioContext.decodeAudioData(res)
      return audioBuffer
    }
    catch {
      return null
    }
  },
  playback: playbackManager,
})

void speechRuntimeStore.registerHost(speechPipeline)

speechPipeline.on('onSpecial', (segment) => {
  if (segment.special)
    playSpecialToken(segment.special)
})

playbackManager.onEnd(({ item }) => {
  if (item.special)
    playSpecialToken(item.special)

  nowSpeaking.value = false
  mouthOpenSize.value = 0
})

playbackManager.onStart(({ item }) => {
  nowSpeaking.value = true
  // NOTICE: postCaption and postPresent may throw errors if the BroadcastChannel is closed
  // (e.g., when navigating away from the page). We wrap these in try-catch to prevent
  // breaking playback when the channel is unavailable.
  assistantCaption.value += ` ${item.text}`
  try {
    postCaption({ type: 'caption-assistant', text: assistantCaption.value })
  }
  catch {
    // BroadcastChannel may be closed - don't break playback
  }
  try {
    postPresent({ type: 'assistant-append', text: item.text })
  }
  catch {
    // BroadcastChannel may be closed - don't break playback
  }
})

function startLipSyncLoop() {
  if (lipSyncLoopId.value)
    return

  const tick = () => {
    if (!nowSpeaking.value || !live2dLipSync.value) {
      mouthOpenSize.value = 0
    }
    else {
      mouthOpenSize.value = live2dLipSync.value.getMouthOpen()
    }
    lipSyncLoopId.value = requestAnimationFrame(tick)
  }

  lipSyncLoopId.value = requestAnimationFrame(tick)
}

async function setupLipSync() {
  if (lipSyncStarted.value)
    return

  try {
    const lipSync = await createLive2DLipSync(audioContext, wlipsyncProfile as Profile, live2dLipSyncOptions)
    live2dLipSync.value = lipSync
    lipSyncNode.value = lipSync.node
    await audioContext.resume()
    startLipSyncLoop()
    lipSyncStarted.value = true
  }
  catch (error) {
    lipSyncStarted.value = false
    console.error('Failed to setup Live2D lip sync', error)
  }
}

function setupAnalyser() {
  if (!audioAnalyser.value) {
    audioAnalyser.value = audioContext.createAnalyser()
  }
}

let currentChatIntent: ReturnType<typeof speechRuntimeStore.openIntent> | null = null

chatHookCleanups.push(onBeforeMessageComposed(async (message) => {
  await updateMemoryContextForPrompt(message)
}))

chatHookCleanups.push(onBeforeMessageComposed(async () => {
  playbackManager.stopAll('new-message')

  setupAnalyser()
  await setupLipSync()
  // Reset assistant caption for a new message
  assistantCaption.value = ''
  try {
    postCaption({ type: 'caption-assistant', text: '' })
  }
  catch (error) {
    // BroadcastChannel may be closed if user navigated away - don't break flow
    console.warn('[Stage] Failed to post caption reset (channel may be closed)', { error })
  }
  try {
    postPresent({ type: 'assistant-reset' })
  }
  catch (error) {
    // BroadcastChannel may be closed if user navigated away - don't break flow
    console.warn('[Stage] Failed to post present reset (channel may be closed)', { error })
  }

  if (currentChatIntent) {
    currentChatIntent.cancel('new-message')
    currentChatIntent = null
  }

  currentChatIntent = speechRuntimeStore.openIntent({
    ownerId: activeCardId.value,
    priority: 'normal',
    behavior: 'queue',
  })
}))

chatHookCleanups.push(onBeforeSend(async () => {
  currentMotion.value = { group: EmotionThinkMotionName }
}))

chatHookCleanups.push(onTokenLiteral(async (literal) => {
  currentChatIntent?.writeLiteral(literal)
}))

chatHookCleanups.push(onTokenSpecial(async (special) => {
  currentChatIntent?.writeSpecial(special)
}))

chatHookCleanups.push(onStreamEnd(async () => {
  delaysQueue.enqueue(llmInferenceEndToken)
  currentChatIntent?.writeFlush()
}))

chatHookCleanups.push(onBeforeSend(async (message, context) => {
  const createdAt = context.message.createdAt ?? Date.now()
  void persistMemoryEntry('user', message, createdAt)
}))

chatHookCleanups.push(onAssistantResponseEnd(async (message) => {
  currentChatIntent?.end()
  currentChatIntent = null
  void persistMemoryEntry('assistant', message, Date.now())
}))

onUnmounted(() => {
  lipSyncStarted.value = false
  embedProvider?.terminateEmbed()
  embedProvider = null
})

// Resume audio context on first user interaction (browser requirement)
let audioContextResumed = false
function resumeAudioContextOnInteraction() {
  if (audioContextResumed || !audioContext)
    return
  audioContextResumed = true
  audioContext.resume().catch(() => {
    // Ignore errors - audio context will be resumed when needed
  })
}

// Add event listeners for user interaction
if (typeof window !== 'undefined') {
  const events = ['click', 'touchstart', 'keydown']
  events.forEach((event) => {
    window.addEventListener(event, resumeAudioContextOnInteraction, { once: true, passive: true })
  })
}

onMounted(async () => {
  if (memoryEnabled.value)
    await initMemoryDb()
})

watch(memoryEnabled, async (enabled) => {
  if (enabled)
    await initMemoryDb()
})

function canvasElement() {
  if (stageModelRenderer.value === 'live2d')
    return live2dSceneRef.value?.canvasElement()

  else if (stageModelRenderer.value === 'vrm')
    return vrmViewerRef.value?.canvasElement()
}

function readRenderTargetRegionAtClientPoint(clientX: number, clientY: number, radius: number) {
  if (stageModelRenderer.value !== 'vrm')
    return null

  return vrmViewerRef.value?.readRenderTargetRegionAtClientPoint?.(clientX, clientY, radius) ?? null
}

onUnmounted(() => {
  if (lipSyncLoopId.value) {
    cancelAnimationFrame(lipSyncLoopId.value)
    lipSyncLoopId.value = undefined
  }

  chatHookCleanups.forEach(dispose => dispose?.())
  viewUpdateCleanups.forEach(dispose => dispose?.())
})

defineExpose({
  canvasElement,
  readRenderTargetRegionAtClientPoint,
})
</script>

<template>
  <div relative>
    <div h-full w-full>
      <Live2DScene
        v-if="stageModelRenderer === 'live2d' && showStage"
        ref="live2dSceneRef"
        v-model:state="componentState"
        min-w="50% <lg:full" min-h="100 sm:100"
        h-full w-full flex-1
        :model-src="stageModelSelectedUrl"
        :model-id="stageModelSelected"
        :focus-at="focusAt"
        :mouth-open-size="mouthOpenSize"
        :paused="paused"
        :x-offset="xOffset"
        :y-offset="yOffset"
        :scale="scale"
        :disable-focus-at="live2dDisableFocus"
        :theme-colors-hue="themeColorsHue"
        :theme-colors-hue-dynamic="themeColorsHueDynamic"
        :live2d-idle-animation-enabled="live2dIdleAnimationEnabled"
        :live2d-auto-blink-enabled="live2dAutoBlinkEnabled"
        :live2d-force-auto-blink-enabled="live2dForceAutoBlinkEnabled"
        :live2d-shadow-enabled="live2dShadowEnabled"
      />
      <ThreeScene
        v-if="stageModelRenderer === 'vrm' && showStage"
        ref="vrmViewerRef"
        v-model:state="componentState"
        :model-src="stageModelSelectedUrl"
        :idle-animation="animations.idleLoop.toString()"
        min-w="50% <lg:full" min-h="100 sm:100" h-full w-full flex-1
        :paused="paused"
        :show-axes="stageViewControlsEnabled"
        :current-audio-source="currentAudioSource"
        @error="console.error"
      />
    </div>
  </div>
</template>
