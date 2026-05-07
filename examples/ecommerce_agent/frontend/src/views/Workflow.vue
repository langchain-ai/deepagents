<template>
  <div class="workflow">
    <div class="workflow-header">
      <h2>{{ task.title }}</h2>
      <div class="task-info">
        <el-tag :type="getStatusType(task.status)">{{ getStatusName(task.status) }}</el-tag>
        <span>店铺: {{ task.store_name }}</span>
        <span>类型: {{ getTypeName(task.type) }}</span>
      </div>
    </div>

    <div class="progress-bar">
      <el-progress :percentage="task.progress" status="success" />
      <div class="progress-label">整体进度: {{ task.progress }}%</div>
    </div>

    <div class="workflow-steps">
      <div 
        v-for="(step, index) in steps" 
        :key="step.id"
        class="step-item"
        :class="{ 
          active: currentStep === index,
          completed: index < currentStep,
          pending: index > currentStep,
          failed: step.status === 'failed'
        }"
      >
        <div class="step-icon">
          <el-icon v-if="index < currentStep"><component :is="icons.CheckCircle" /></el-icon>
          <el-icon v-else-if="index === currentStep"><component :is="icons.Loading" /></el-icon>
          <span v-else>{{ index + 1 }}</span>
        </div>
        <div class="step-content">
          <h4 class="step-title">{{ step.title }}</h4>
          <p class="step-desc">{{ step.description }}</p>
          <div class="step-status">
            <span :class="step.status">{{ getStepStatusName(step.status) }}</span>
            <span v-if="step.time" class="step-time">{{ step.time }}</span>
          </div>
          <div v-if="step.logs && step.logs.length" class="step-logs">
            <div v-for="(log, logIndex) in step.logs" :key="logIndex" class="log-item">
              <span class="log-time">{{ log.time }}</span>
              <span :class="log.type">{{ log.message }}</span>
            </div>
          </div>
        </div>
        <div v-if="index < steps.length - 1" class="step-line"></div>
      </div>
    </div>

    <div class="workflow-actions">
      <el-button 
        v-if="task.status === 'running'" 
        @click="pauseTask"
        icon="Pause"
      >暂停任务</el-button>
      <el-button 
        v-if="task.status === 'pending'" 
        type="primary"
        @click="startTask"
        icon="Play"
      >启动任务</el-button>
      <el-button 
        v-if="task.status === 'failed'" 
        type="success"
        @click="retryTask"
        icon="Refresh"
      >重试任务</el-button>
      <el-button 
        v-if="task.status === 'completed'" 
        type="primary"
        @click="createNewTask"
        icon="Plus"
      >创建新任务</el-button>
      <el-button 
        type="danger" 
        @click="cancelTask"
        icon="Delete"
      >取消任务</el-button>
    </div>

    <div class="task-logs">
      <h3>任务日志</h3>
      <el-scrollbar height="200px" class="logs-scroll">
        <div v-for="(log, index) in taskLogs" :key="index" class="log-row">
          <span class="log-timestamp">{{ log.timestamp }}</span>
          <span :class="log.level">{{ log.level }}</span>
          <span class="log-message">{{ log.message }}</span>
        </div>
      </el-scrollbar>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import * as icons from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()

const taskId = parseInt(route.params.taskId as string)

const task = ref({
  id: taskId,
  title: '发布新款商品',
  type: 'publish',
  status: 'running',
  store_name: '抖音官方旗舰店',
  progress: 60,
  created_at: '2024-03-01 10:00',
  started_at: '2024-03-01 10:05'
})

const currentStep = ref(2)

const steps = ref([
  {
    id: 1,
    title: '第一步: 登录平台',
    description: '使用店铺账号登录目标电商平台',
    status: 'completed',
    time: '10:05',
    logs: [
      { time: '10:05:02', type: 'info', message: '正在打开浏览器...' },
      { time: '10:05:08', type: 'info', message: '导航到登录页面' },
      { time: '10:05:15', type: 'success', message: '输入账号密码' },
      { time: '10:05:22', type: 'success', message: '登录成功' }
    ]
  },
  {
    id: 2,
    title: '第二步: 进入商品发布页',
    description: '导航到商品发布页面并准备商品数据',
    status: 'completed',
    time: '10:06',
    logs: [
      { time: '10:06:01', type: 'info', message: '进入商家后台' },
      { time: '10:06:08', type: 'info', message: '点击商品管理' },
      { time: '10:06:15', type: 'success', message: '进入发布页面' }
    ]
  },
  {
    id: 3,
    title: '第三步: 填写商品信息',
    description: '填写商品标题、描述、价格等信息',
    status: 'running',
    time: '10:07',
    logs: [
      { time: '10:07:02', type: 'info', message: '正在填写商品标题...' },
      { time: '10:07:15', type: 'info', message: '填写商品描述...' },
      { time: '10:07:30', type: 'info', message: '设置商品价格...' }
    ]
  },
  {
    id: 4,
    title: '第四步: 上传商品图片',
    description: '上传商品主图和详情图片',
    status: 'pending',
    logs: []
  },
  {
    id: 5,
    title: '第五步: 提交审核',
    description: '提交商品信息等待平台审核',
    status: 'pending',
    logs: []
  }
])

const taskLogs = ref([
  { timestamp: '10:05:00', level: 'info', message: '任务开始执行' },
  { timestamp: '10:05:02', level: 'info', message: '正在初始化浏览器环境' },
  { timestamp: '10:05:08', level: 'success', message: '浏览器启动成功' },
  { timestamp: '10:05:15', level: 'info', message: '开始执行第一步: 登录平台' },
  { timestamp: '10:05:22', level: 'success', message: '登录成功' },
  { timestamp: '10:06:00', level: 'info', message: '开始执行第二步: 进入商品发布页' },
  { timestamp: '10:06:15', level: 'success', message: '进入发布页面成功' },
  { timestamp: '10:07:00', level: 'info', message: '开始执行第三步: 填写商品信息' },
  { timestamp: '10:07:30', level: 'info', message: '正在填写商品信息...' }
])

const getTypeName = (type: string): string => {
  const names: Record<string, string> = {
    publish: '商品发布',
    review: '好评管理',
    fetch: '数据采集'
  }
  return names[type] || type
}

const getStatusName = (status: string): string => {
  const names: Record<string, string> = {
    pending: '等待中',
    running: '运行中',
    completed: '已完成',
    failed: '失败'
  }
  return names[status] || status
}

const getStatusType = (status: string): string => {
  const types: Record<string, string> = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return types[status] || 'info'
}

const getStepStatusName = (status: string): string => {
  const names: Record<string, string> = {
    pending: '等待执行',
    running: '执行中',
    completed: '已完成',
    failed: '执行失败'
  }
  return names[status] || status
}

const startTask = () => {
  task.value.status = 'running'
  currentStep.value = 0
}

const pauseTask = () => {
  task.value.status = 'pending'
}

const retryTask = () => {
  task.value.status = 'running'
  task.value.progress = 0
  currentStep.value = 0
  steps.value.forEach(step => {
    step.status = 'pending'
    step.logs = []
  })
}

const cancelTask = () => {
  if (confirm('确定取消任务吗？')) {
    router.push('/tasks')
  }
}

const createNewTask = () => {
  router.push('/tasks')
}

onMounted(() => {
  if (task.value.status === 'running') {
    simulateProgress()
  }
})

const simulateProgress = () => {
  const interval = setInterval(() => {
    if (task.value.progress < 100) {
      task.value.progress += Math.random() * 10
      if (task.value.progress > 100) task.value.progress = 100
      
      if (currentStep.value < steps.value.length && 
          task.value.progress >= ((currentStep.value + 1) * 20)) {
        steps.value[currentStep.value].status = 'completed'
        currentStep.value++
        if (currentStep.value < steps.value.length) {
          steps.value[currentStep.value].status = 'running'
        }
      }
      
      if (task.value.progress >= 100) {
        task.value.status = 'completed'
        clearInterval(interval)
      }
    }
  }, 2000)
}
</script>

<style scoped>
.workflow {
  padding: 20px;
}

.workflow-header {
  margin-bottom: 20px;
}

.workflow-header h2 {
  margin: 0 0 10px 0;
  font-size: 24px;
}

.task-info {
  display: flex;
  gap: 20px;
  align-items: center;
}

.task-info span {
  color: #6b7280;
}

.progress-bar {
  margin-bottom: 30px;
}

.progress-label {
  text-align: right;
  margin-top: 5px;
  font-size: 14px;
  color: #6b7280;
}

.workflow-steps {
  position: relative;
  margin-bottom: 30px;
}

.step-item {
  display: flex;
  align-items: flex-start;
  margin-bottom: 20px;
  position: relative;
}

.step-item:last-child {
  margin-bottom: 0;
}

.step-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: #e5e7eb;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: bold;
  color: #6b7280;
  flex-shrink: 0;
  margin-right: 20px;
  transition: all 0.3s;
}

.step-item.active .step-icon {
  background-color: #3b82f6;
  color: white;
}

.step-item.completed .step-icon {
  background-color: #22c55e;
  color: white;
}

.step-item.failed .step-icon {
  background-color: #ef4444;
  color: white;
}

.step-content {
  flex: 1;
  background-color: #f9fafb;
  border-radius: 8px;
  padding: 15px;
  transition: all 0.3s;
}

.step-item.active .step-content {
  background-color: #dbeafe;
  border: 1px solid #3b82f6;
}

.step-item.completed .step-content {
  background-color: #dcfce7;
}

.step-item.failed .step-content {
  background-color: #fecaca;
}

.step-title {
  margin: 0 0 5px 0;
  font-size: 16px;
  color: #1f2937;
}

.step-desc {
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #6b7280;
}

.step-status {
  display: flex;
  gap: 10px;
  align-items: center;
}

.step-status span:first-child {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.step-status .pending {
  background-color: #e5e7eb;
  color: #6b7280;
}

.step-status .running {
  background-color: #fef3c7;
  color: #d97706;
}

.step-status .completed {
  background-color: #dcfce7;
  color: #16a34a;
}

.step-status .failed {
  background-color: #fecaca;
  color: #dc2626;
}

.step-time {
  font-size: 12px;
  color: #9ca3af;
}

.step-logs {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid #e5e7eb;
}

.log-item {
  display: flex;
  gap: 10px;
  font-size: 12px;
  margin-bottom: 5px;
}

.log-item:last-child {
  margin-bottom: 0;
}

.log-time {
  color: #9ca3af;
}

.log-item .info {
  color: #3b82f6;
}

.log-item .success {
  color: #22c55e;
}

.log-item .error {
  color: #ef4444;
}

.step-line {
  position: absolute;
  left: 19px;
  top: 50px;
  width: 2px;
  height: calc(100% - 30px);
  background-color: #e5e7eb;
}

.workflow-actions {
  display: flex;
  gap: 10px;
  margin-bottom: 30px;
}

.task-logs {
  background-color: #f9fafb;
  border-radius: 8px;
  padding: 15px;
}

.task-logs h3 {
  margin: 0 0 15px 0;
  font-size: 16px;
}

.logs-scroll {
  border-radius: 8px;
}

.log-row {
  display: flex;
  gap: 15px;
  padding: 8px 0;
  border-bottom: 1px solid #e5e7eb;
  font-size: 13px;
}

.log-row:last-child {
  border-bottom: none;
}

.log-timestamp {
  color: #9ca3af;
  width: 100px;
}

.log-row .info {
  color: #3b82f6;
  width: 60px;
}

.log-row .success {
  color: #22c55e;
  width: 60px;
}

.log-row .warning {
  color: #d97706;
  width: 60px;
}

.log-row .error {
  color: #ef4444;
  width: 60px;
}

.log-message {
  flex: 1;
  color: #1f2937;
}
</style>
