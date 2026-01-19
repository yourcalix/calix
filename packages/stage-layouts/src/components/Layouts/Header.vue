<script setup lang="ts">
import HeaderAvatar from './HeaderAvatar.vue'
import HeaderLink from './HeaderLink.vue'
</script>

<template>
  <header mb-1 w-full flex items-center justify-between gap-2>
    <HeaderLink />
    <div flex items-center gap-2>
      <HeaderAvatar />
    </div>
  </header>
</template>
