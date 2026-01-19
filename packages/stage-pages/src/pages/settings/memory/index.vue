<script setup lang="ts">
import { useSettings } from '@proj-airi/stage-ui/stores/settings'
import { FieldCheckbox } from '@proj-airi/ui'
import { useI18n } from 'vue-i18n'

const settings = useSettings()
const { t } = useI18n()
</script>

<template>
  <div rounded-lg bg-neutral-50 p-4 dark:bg-neutral-800 flex="~ col gap-4">
    <FieldCheckbox
      v-model="settings.memoryEnabled"
      v-motion
      mb-2
      :initial="{ opacity: 0, y: 10 }"
      :enter="{ opacity: 1, y: 0 }"
      :duration="250 + (2 * 10)"
      :delay="2 * 50"
      transition="all ease-in-out duration-250"
      :label="t('settings.pages.memory.title')"
      :description="t('settings.pages.memory.description')"
    />
  </div>
</template>

<route lang="yaml">
meta:
  layout: settings
  titleKey: settings.pages.memory.title
  subtitleKey: settings.title
  descriptionKey: settings.pages.memory.description
  icon: i-solar:leaf-bold-duotone
  settingsEntry: true
  order: 5
  stageTransition:
    name: slide
</route>
