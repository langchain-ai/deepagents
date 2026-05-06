<template>
  <div class="tasks">
    <el-card>
      <h2>任务管理</h2>
      <el-button type="primary" @click="showAddDialog = true" style="margin-bottom: 20px;">
        创建任务
      </el-button>
      <el-table :data="tasks" style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="name" label="任务名称" width="200" />
        <el-table-column prop="task_type" label="类型" width="120">
          <template #default="{ row }">
            <el-tag>{{ row.task_type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="progress" label="进度" width="150">
          <template #default="{ row }">
            <el-progress :percentage="row.progress" />
          </template>
        </el-table-column>
        <el-table-column prop="current_step" label="当前步骤" show-overflow-tooltip />
        <el-table-column prop="created_at" label="创建时间" />
      </el-table>
    </el-card>

    <el-dialog v-model="showAddDialog" title="创建任务" width="500">
      <el-form :model="newTask" label-width="100px">
        <el-form-item label="店铺">
          <el-select v-model="newTask.store_id" placeholder="请选择店铺">
            <el-option 
              v-for="store in stores" 
              :key="store.id" 
              :label="store.name" 
              :value="store.id" 
            />
          </el-select>
        </el-form-item>
        <el-form-item label="任务类型">
          <el-select v-model="newTask.task_type" placeholder="请选择类型">
            <el-option label="商品发布" value="publish" />
            <el-option label="好评管理" value="good_review" />
            <el-option label="数据采集" value="fetch_data" />
            <el-option label="运营分析" value="analyze" />
          </el-select>
        </el-form-item>
        <el-form-item label="任务名称">
          <el-input v-model="newTask.name" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddDialog = false">取消</el-button>
        <el-button type="primary" @click="createTask">创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'

const tasks = ref<any[]>([])
const stores = ref<any[]>([])
const showAddDialog = ref(false)
const newTask = ref({
  store_id: null as number | null,
  task_type: '',
  name: ''
})

const loadTasks = async () => {
  try {
    const res = await axios.get('/api/tasks')
    tasks.value = res.data
  } catch (error) {
    console.error('Failed to load tasks:', error)
  }
}

const loadStores = async () => {
  try {
    const res = await axios.get('/api/stores')
    stores.value = res.data
  } catch (error) {
    console.error('Failed to load stores:', error)
  }
}

const createTask = async () => {
  try {
    await axios.post('/api/tasks', null, {
      params: newTask.value
    })
    showAddDialog.value = false
    await loadTasks()
  } catch (error) {
    console.error('Failed to create task:', error)
  }
}

const getStatusType = (status: string) => {
  const map: Record<string, any> = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return map[status] || ''
}

const getStatusText = (status: string) => {
  const map: Record<string, string> = {
    pending: '待执行',
    running: '运行中',
    completed: '已完成',
    failed: '失败'
  }
  return map[status] || status
}

onMounted(() => {
  loadTasks()
  loadStores()
})
</script>
