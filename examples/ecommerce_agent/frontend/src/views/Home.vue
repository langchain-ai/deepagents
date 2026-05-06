<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card>
          <h2>欢迎使用电商自动化助手</h2>
          <p>基于 DeepAgents 的多平台电商自动化运营工具</p>
        </el-card>
      </el-col>
    </el-row>
    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="8">
        <el-card>
          <div class="stat-item">
            <h3>店铺</h3>
            <p class="stat-number">{{ storeCount }}</p>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <div class="stat-item">
            <h3>任务</h3>
            <p class="stat-number">{{ taskCount }}</p>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <div class="stat-item">
            <h3>运行中</h3>
            <p class="stat-number">{{ runningCount }}</p>
          </div>
        </el-card>
      </el-col>
    </el-row>
    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="24">
        <el-card>
          <h3>快捷操作</h3>
          <el-button type="primary" @click="$router.push('/stores')">
            管理店铺
          </el-button>
          <el-button type="success" @click="$router.push('/tasks')">
            查看任务
          </el-button>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'

const storeCount = ref(0)
const taskCount = ref(0)
const runningCount = ref(0)

onMounted(async () => {
  try {
    const [storesRes, tasksRes] = await Promise.all([
      axios.get('/api/stores'),
      axios.get('/api/tasks')
    ])
    storeCount.value = storesRes.data.length
    taskCount.value = tasksRes.data.length
    runningCount.value = tasksRes.data.filter((t: any) => t.status === 'running').length
  } catch (error) {
    console.error('Failed to load stats:', error)
  }
})
</script>

<style scoped>
.stat-item {
  text-align: center;
}

.stat-number {
  font-size: 36px;
  font-weight: bold;
  color: #409eff;
}
</style>
