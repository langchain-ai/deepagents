<template>
  <div class="tasks">
    <div class="toolbar">
      <el-button type="primary" @click="showCreateModal = true" icon="Plus">创建任务</el-button>
      <el-select v-model="filterStatus" placeholder="筛选状态">
        <el-option label="全部" value="" />
        <el-option label="等待中" value="pending" />
        <el-option label="运行中" value="running" />
        <el-option label="已完成" value="completed" />
        <el-option label="失败" value="failed" />
      </el-select>
      <el-select v-model="filterType" placeholder="筛选类型">
        <el-option label="全部" value="" />
        <el-option label="商品发布" value="publish" />
        <el-option label="好评管理" value="review" />
        <el-option label="数据采集" value="fetch" />
      </el-select>
    </div>

    <div class="tasks-grid">
      <el-card 
        v-for="task in filteredTasks" 
        :key="task.id" 
        class="task-card"
        @click="goToWorkflow(task.id)"
      >
        <div class="task-header">
          <div class="task-type" :class="task.type">
            {{ getTypeName(task.type) }}
          </div>
          <el-tag :type="getStatusType(task.status)">
            {{ getStatusName(task.status) }}
          </el-tag>
        </div>
        <h3 class="task-title">{{ task.title }}</h3>
        <div class="task-info">
          <div>目标店铺: {{ task.store_name }}</div>
          <div>创建时间: {{ task.created_at }}</div>
          <div v-if="task.started_at">开始时间: {{ task.started_at }}</div>
          <div v-if="task.completed_at">完成时间: {{ task.completed_at }}</div>
        </div>
        <div class="task-progress" v-if="task.status === 'running'">
          <el-progress :percentage="task.progress" :show-text="false" />
          <span class="progress-text">{{ task.progress }}%</span>
        </div>
        <div class="task-actions">
          <el-button 
            v-if="task.status === 'running'" 
            size="small" 
            @click.stop="pauseTask(task)"
            icon="Pause"
          >暂停</el-button>
          <el-button 
            v-if="task.status === 'pending'" 
            size="small" 
            type="primary"
            @click.stop="startTask(task)"
            icon="Play"
          >启动</el-button>
          <el-button 
            v-if="task.status === 'failed'" 
            size="small" 
            type="success"
            @click.stop="retryTask(task)"
            icon="Refresh"
          >重试</el-button>
          <el-button 
            size="small" 
            type="danger" 
            @click.stop="deleteTask(task)"
            icon="Delete"
          >删除</el-button>
        </div>
      </el-card>
    </div>

    <el-dialog title="创建任务" :visible.sync="showCreateModal">
      <el-form :model="form" label-width="100px">
        <el-form-item label="任务名称">
          <el-input v-model="form.title" />
        </el-form-item>
        <el-form-item label="任务类型">
          <el-select v-model="form.type">
            <el-option label="商品发布" value="publish" />
            <el-option label="好评管理" value="review" />
            <el-option label="数据采集" value="fetch" />
          </el-select>
        </el-form-item>
        <el-form-item label="目标店铺">
          <el-select v-model="form.store_id">
            <el-option v-for="store in stores" :key="store.id" :label="store.name" :value="store.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="定时执行">
          <el-switch v-model="form.scheduled" />
        </el-form-item>
        <el-form-item label="执行时间" v-if="form.scheduled">
          <el-time-picker v-model="form.schedule_time" format="HH:mm" />
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showCreateModal = false">取消</el-button>
        <el-button type="primary" @click="createTask">创建</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

interface Task {
  id: number
  title: string
  type: string
  status: string
  store_id: number
  store_name: string
  progress: number
  created_at: string
  started_at?: string
  completed_at?: string
}

const tasks = ref<Task[]>([
  { id: 1, title: '发布新款商品', type: 'publish', status: 'running', store_id: 1, store_name: '抖音官方旗舰店', progress: 60, created_at: '2024-03-01 10:00', started_at: '2024-03-01 10:05' },
  { id: 2, title: '处理用户好评', type: 'review', status: 'completed', store_id: 2, store_name: '拼多多专营店', progress: 100, created_at: '2024-03-01 09:00', started_at: '2024-03-01 09:05', completed_at: '2024-03-01 09:30' },
  { id: 3, title: '采集竞品数据', type: 'fetch', status: 'pending', store_id: 3, store_name: '淘宝皇冠店', progress: 0, created_at: '2024-03-01 08:00' },
  { id: 4, title: '批量发布商品', type: 'publish', status: 'failed', store_id: 1, store_name: '抖音官方旗舰店', progress: 30, created_at: '2024-02-28 14:00', started_at: '2024-02-28 14:05' },
  { id: 5, title: '同步订单数据', type: 'fetch', status: 'completed', store_id: 4, store_name: '京东自营店', progress: 100, created_at: '2024-02-28 10:00', started_at: '2024-02-28 10:10', completed_at: '2024-02-28 10:45' },
  { id: 6, title: '商品信息更新', type: 'publish', status: 'pending', store_id: 5, store_name: '小红书种草店', progress: 0, created_at: '2024-03-01 11:00' }
])

const stores = ref([
  { id: 1, name: '抖音官方旗舰店' },
  { id: 2, name: '拼多多专营店' },
  { id: 3, name: '淘宝皇冠店' },
  { id: 4, name: '京东自营店' },
  { id: 5, name: '小红书种草店' }
])

const showCreateModal = ref(false)
const filterStatus = ref('')
const filterType = ref('')

const form = ref({
  title: '',
  type: 'publish',
  store_id: 1,
  scheduled: false,
  schedule_time: ''
})

const filteredTasks = computed(() => {
  return tasks.value.filter(task => {
    if (filterStatus.value && task.status !== filterStatus.value) return false
    if (filterType.value && task.type !== filterType.value) return false
    return true
  })
})

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

const goToWorkflow = (taskId: number) => {
  router.push(`/workflow/${taskId}`)
}

const createTask = () => {
  const store = stores.value.find(s => s.id === form.value.store_id)
  tasks.value.push({
    id: tasks.value.length + 1,
    title: form.value.title,
    type: form.value.type,
    status: 'pending',
    store_id: form.value.store_id,
    store_name: store?.name || '',
    progress: 0,
    created_at: new Date().toLocaleString('zh-CN')
  })
  showCreateModal.value = false
  form.value = { title: '', type: 'publish', store_id: 1, scheduled: false, schedule_time: '' }
}

const startTask = (task: Task) => {
  task.status = 'running'
  task.started_at = new Date().toLocaleString('zh-CN')
}

const pauseTask = (task: Task) => {
  task.status = 'pending'
}

const retryTask = (task: Task) => {
  task.status = 'running'
  task.progress = 0
  task.started_at = new Date().toLocaleString('zh-CN')
}

const deleteTask = (task: Task) => {
  if (confirm(`确定删除任务 "${task.title}" 吗？`)) {
    tasks.value = tasks.value.filter(t => t.id !== task.id)
  }
}
</script>

<style scoped>
.tasks {
  padding: 10px;
}

.toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
}

.toolbar :deep(.el-select) {
  width: 150px;
}

.tasks-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.task-card {
  cursor: pointer;
  transition: all 0.3s;
}

.task-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.task-type {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
}

.task-type.publish {
  background-color: #dbeafe;
  color: #3b82f6;
}

.task-type.review {
  background-color: #dcfce7;
  color: #22c55e;
}

.task-type.fetch {
  background-color: #fef3c7;
  color: #f59e0b;
}

.task-title {
  margin: 0 0 10px 0;
  font-size: 16px;
  color: #1f2937;
}

.task-info {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 15px;
}

.task-info div {
  margin-bottom: 4px;
}

.task-progress {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
}

.progress-text {
  font-size: 14px;
  font-weight: 600;
}

.task-actions {
  display: flex;
  gap: 8px;
}
</style>
