<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon tasks">
            <el-icon><component :is="icons.Task" /></el-icon>
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ stats.tasks }}</div>
            <div class="stat-label">今日任务</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon success">
            <el-icon><component :is="icons.CheckCircle" /></el-icon>
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ stats.success }}</div>
            <div class="stat-label">完成任务</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon stores">
            <el-icon><component :is="icons.Store" /></el-icon>
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ stats.stores }}</div>
            <div class="stat-label">管理店铺</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon alerts">
            <el-icon><component :is="icons.Bell" /></el-icon>
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ stats.alerts }}</div>
            <div class="stat-label">待处理告警</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="12">
        <el-card title="最近任务" class="task-list-card">
          <el-timeline>
            <el-timeline-item
              v-for="task in recentTasks"
              :key="task.id"
              :timestamp="task.time"
              :type="task.type"
            >
              <div class="timeline-content">
                <div class="task-title">{{ task.title }}</div>
                <div class="task-desc">{{ task.desc }}</div>
              </div>
            </el-timeline-item>
          </el-timeline>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card title="系统状态" class="system-status-card">
          <el-table :data="systemStatus" border>
            <el-table-column prop="name" label="服务" />
            <el-table-column prop="status" label="状态">
              <template #default="scope">
                <el-tag :type="scope.row.status === '运行中' ? 'success' : 'danger'">
                  {{ scope.row.status }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="cpu" label="CPU" />
            <el-table-column prop="memory" label="内存" />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import * as icons from '@element-plus/icons-vue'

const stats = ref({
  tasks: 12,
  success: 8,
  stores: 5,
  alerts: 2
})

const recentTasks = ref([
  { id: 1, title: '商品发布任务', desc: '成功发布3件商品到抖音', time: '10分钟前', type: 'success' },
  { id: 2, title: '好评管理任务', desc: '处理了5条好评', time: '30分钟前', type: 'primary' },
  { id: 3, title: '数据采集任务', desc: '采集淘宝商品数据', time: '1小时前', type: 'info' },
  { id: 4, title: '订单同步任务', desc: '同步拼多多订单', time: '2小时前', type: 'warning' },
  { id: 5, title: '商品发布任务', desc: '发布失败，需要重试', time: '3小时前', type: 'danger' }
])

const systemStatus = ref([
  { name: '浏览器服务', status: '运行中', cpu: '15%', memory: '256MB' },
  { name: '任务调度', status: '运行中', cpu: '5%', memory: '128MB' },
  { name: '数据库', status: '运行中', cpu: '10%', memory: '512MB' },
  { name: '向量存储', status: '运行中', cpu: '8%', memory: '256MB' },
  { name: 'API服务', status: '运行中', cpu: '12%', memory: '384MB' }
])
</script>

<style scoped>
.home {
  padding: 10px;
}

.stat-card {
  display: flex;
  align-items: center;
  padding: 20px;
}

.stat-icon {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  margin-right: 20px;
}

.stat-icon.tasks {
  background-color: #dbeafe;
  color: #3b82f6;
}

.stat-icon.success {
  background-color: #dcfce7;
  color: #22c55e;
}

.stat-icon.stores {
  background-color: #fef3c7;
  color: #f59e0b;
}

.stat-icon.alerts {
  background-color: #fecaca;
  color: #ef4444;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: #1f2937;
}

.stat-label {
  font-size: 14px;
  color: #6b7280;
}

.task-list-card,
.system-status-card {
  height: 350px;
}

.timeline-content {
  margin-top: 5px;
}

.task-title {
  font-weight: 600;
  color: #1f2937;
}

.task-desc {
  font-size: 12px;
  color: #6b7280;
  margin-top: 5px;
}
</style>
