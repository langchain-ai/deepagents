<template>
  <div class="scheduled-tasks">
    <div class="toolbar">
      <el-button type="primary" @click="showAddModal = true" icon="Plus">添加定时任务</el-button>
      <el-select v-model="filterStatus" placeholder="筛选状态">
        <el-option label="全部" value="" />
        <el-option label="运行中" value="active" />
        <el-option label="已暂停" value="paused" />
      </el-select>
    </div>

    <el-card class="task-list-card">
      <el-table :data="tasks" border>
        <el-table-column prop="name" label="任务名称" />
        <el-table-column prop="task_type" label="任务类型">
          <template #default="scope">
            <el-tag :type="getTaskTypeTag(scope.row.task_type)">
              {{ getTaskTypeName(scope.row.task_type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="store_name" label="目标店铺" />
        <el-table-column prop="cron_expression" label="执行时间">
          <template #default="scope">
            <div class="cron-display">{{ formatCron(scope.row.cron_expression) }}</div>
          </template>
        </el-table-column>
        <el-table-column prop="last_run" label="上次执行" />
        <el-table-column prop="next_run" label="下次执行" />
        <el-table-column prop="status" label="状态">
          <template #default="scope">
            <el-tag :type="scope.row.status === 'active' ? 'success' : 'warning'">
              {{ scope.row.status === 'active' ? '运行中' : '已暂停' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作">
          <template #default="scope">
            <el-button 
              v-if="scope.row.status === 'active'" 
              size="small" 
              @click="pauseTask(scope.row)"
              icon="Pause"
            >暂停</el-button>
            <el-button 
              v-if="scope.row.status === 'paused'" 
              size="small" 
              type="success"
              @click="resumeTask(scope.row)"
              icon="Play"
            >恢复</el-button>
            <el-button size="small" @click="editTask(scope.row)" icon="Edit">编辑</el-button>
            <el-button size="small" type="danger" @click="deleteTask(scope.row)" icon="Delete">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog :title="isEditing ? '编辑定时任务' : '添加定时任务'" :visible.sync="showAddModal">
      <el-form :model="form" label-width="100px">
        <el-form-item label="任务名称" required>
          <el-input v-model="form.name" />
        </el-form-item>
        <el-form-item label="任务类型" required>
          <el-select v-model="form.task_type">
            <el-option label="商品发布" value="publish" />
            <el-option label="好评管理" value="review" />
            <el-option label="数据采集" value="fetch" />
            <el-option label="订单同步" value="order_sync" />
          </el-select>
        </el-form-item>
        <el-form-item label="目标店铺" required>
          <el-select v-model="form.store_id">
            <el-option v-for="store in stores" :key="store.id" :label="store.name" :value="store.id" />
          </el-select>
        </el-form-item>
        <el-form-item label="执行频率">
          <el-select v-model="form.frequency" @change="updateCronExpression">
            <el-option label="每天" value="daily" />
            <el-option label="每周" value="weekly" />
            <el-option label="每月" value="monthly" />
            <el-option label="自定义" value="custom" />
          </el-select>
        </el-form-item>
        <el-form-item label="执行时间" v-if="form.frequency !== 'custom'">
          <el-time-picker v-model="form.time" format="HH:mm" />
        </el-form-item>
        <el-form-item label="Cron表达式" v-if="form.frequency === 'custom'">
          <el-input v-model="form.cron_expression" placeholder="如: 0 30 9 * * ?" />
        </el-form-item>
        <el-form-item label="备注">
          <el-input v-model="form.remark" type="textarea" />
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showAddModal = false">取消</el-button>
        <el-button type="primary" @click="saveTask">保存</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface ScheduledTask {
  id: number
  name: string
  task_type: string
  store_id: number
  store_name: string
  cron_expression: string
  last_run: string
  next_run: string
  status: string
  remark?: string
}

const stores = ref([
  { id: 1, name: '抖音官方旗舰店' },
  { id: 2, name: '拼多多专营店' },
  { id: 3, name: '淘宝皇冠店' },
  { id: 4, name: '京东自营店' },
  { id: 5, name: '小红书种草店' }
])

const tasks = ref<ScheduledTask[]>([
  { id: 1, name: '每日商品发布', task_type: 'publish', store_id: 1, store_name: '抖音官方旗舰店', cron_expression: '0 0 9 * * ?', last_run: '2024-03-01 09:00:00', next_run: '2024-03-02 09:00:00', status: 'active', remark: '每天早上9点发布新品' },
  { id: 2, name: '每日好评处理', task_type: 'review', store_id: 2, store_name: '拼多多专营店', cron_expression: '0 0 10 * * ?', last_run: '2024-03-01 10:00:00', next_run: '2024-03-02 10:00:00', status: 'active', remark: '每天上午10点处理好评' },
  { id: 3, name: '竞品数据采集', task_type: 'fetch', store_id: 3, store_name: '淘宝皇冠店', cron_expression: '0 0 8 * * ?', last_run: '2024-03-01 08:00:00', next_run: '2024-03-02 08:00:00', status: 'active' },
  { id: 4, name: '订单同步', task_type: 'order_sync', store_id: 4, store_name: '京东自营店', cron_expression: '0 0/30 * * * ?', last_run: '2024-03-01 11:30:00', next_run: '2024-03-01 12:00:00', status: 'active', remark: '每30分钟同步一次订单' },
  { id: 5, name: '周报数据采集', task_type: 'fetch', store_id: 1, store_name: '抖音官方旗舰店', cron_expression: '0 0 10 * * MON', last_run: '2024-02-26 10:00:00', next_run: '2024-03-04 10:00:00', status: 'paused' }
])

const showAddModal = ref(false)
const isEditing = ref(false)
const filterStatus = ref('')

const form = ref({
  id: 0,
  name: '',
  task_type: 'publish',
  store_id: 1,
  frequency: 'daily',
  time: '09:00',
  cron_expression: '0 0 9 * * ?',
  remark: ''
})

const getTaskTypeName = (type: string): string => {
  const names: Record<string, string> = {
    publish: '商品发布',
    review: '好评管理',
    fetch: '数据采集',
    order_sync: '订单同步'
  }
  return names[type] || type
}

const getTaskTypeTag = (type: string): string => {
  const types: Record<string, string> = {
    publish: 'primary',
    review: 'success',
    fetch: 'warning',
    order_sync: 'info'
  }
  return types[type] || 'info'
}

const formatCron = (cron: string): string => {
  const parts = cron.split(' ')
  if (parts.length >= 5) {
    const minute = parts[0]
    const hour = parts[1]
    const day = parts[2]
    const month = parts[3]
    const week = parts[4]
    
    if (week !== '?' && day === '?') {
      const weekDays = ['日', '一', '二', '三', '四', '五', '六']
      const weekNum = parseInt(week)
      if (!isNaN(weekNum) && weekNum >= 1 && weekNum <= 7) {
        return `${hour}:${minute} 每周${weekDays[weekNum - 1]}`
      }
    }
    
    if (day !== '?' && month === '*' && week === '?') {
      return `${hour}:${minute} 每天`
    }
    
    if (day !== '?' && month !== '*' && week === '?') {
      return `${hour}:${minute} 每月${day}日`
    }
  }
  return cron
}

const updateCronExpression = () => {
  const time = form.value.time ? form.value.time.split(':') : ['09', '00']
  const hour = time[0]
  const minute = time[1]
  
  switch (form.value.frequency) {
    case 'daily':
      form.value.cron_expression = `0 ${minute} ${hour} * * ?`
      break
    case 'weekly':
      form.value.cron_expression = `0 ${minute} ${hour} ? * MON`
      break
    case 'monthly':
      form.value.cron_expression = `0 ${minute} ${hour} 1 * ?`
      break
  }
}

const editTask = (task: ScheduledTask) => {
  isEditing.value = true
  form.value = {
    id: task.id,
    name: task.name,
    task_type: task.task_type,
    store_id: task.store_id,
    frequency: 'custom',
    time: '',
    cron_expression: task.cron_expression,
    remark: task.remark || ''
  }
  showAddModal.value = true
}

const saveTask = () => {
  const store = stores.value.find(s => s.id === form.value.store_id)
  
  if (isEditing.value) {
    const index = tasks.value.findIndex(t => t.id === form.value.id)
    if (index !== -1) {
      tasks.value[index] = {
        ...tasks.value[index],
        name: form.value.name,
        task_type: form.value.task_type,
        store_id: form.value.store_id,
        store_name: store?.name || '',
        cron_expression: form.value.cron_expression,
        remark: form.value.remark
      }
    }
  } else {
    tasks.value.push({
      id: tasks.value.length + 1,
      name: form.value.name,
      task_type: form.value.task_type,
      store_id: form.value.store_id,
      store_name: store?.name || '',
      cron_expression: form.value.cron_expression,
      last_run: '-',
      next_run: new Date().toLocaleString('zh-CN'),
      status: 'active',
      remark: form.value.remark
    })
  }
  
  showAddModal.value = false
  form.value = { id: 0, name: '', task_type: 'publish', store_id: 1, frequency: 'daily', time: '09:00', cron_expression: '0 0 9 * * ?', remark: '' }
  isEditing.value = false
}

const pauseTask = (task: ScheduledTask) => {
  task.status = 'paused'
}

const resumeTask = (task: ScheduledTask) => {
  task.status = 'active'
}

const deleteTask = (task: ScheduledTask) => {
  if (confirm(`确定删除定时任务 "${task.name}" 吗？`)) {
    tasks.value = tasks.value.filter(t => t.id !== task.id)
  }
}
</script>

<style scoped>
.scheduled-tasks {
  padding: 10px;
}

.toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 20px;
}

.toolbar :deep(.el-select) {
  width: 150px;
}

.task-list-card {
  margin-top: 20px;
}

.cron-display {
  font-size: 13px;
  color: #374151;
}
</style>
