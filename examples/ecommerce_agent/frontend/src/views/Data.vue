<template>
  <div class="data">
    <div class="toolbar">
      <el-select v-model="selectedStore" placeholder="选择店铺">
        <el-option label="全部店铺" value="" />
        <el-option v-for="store in stores" :key="store.id" :label="store.name" :value="store.id" />
      </el-select>
      <el-select v-model="timeRange" placeholder="时间范围">
        <el-option label="今日" value="today" />
        <el-option label="本周" value="week" />
        <el-option label="本月" value="month" />
        <el-option label="本季度" value="quarter" />
      </el-select>
      <el-button type="primary" @click="refreshData" icon="Refresh">刷新数据</el-button>
    </div>

    <el-row :gutter="20" style="margin-bottom: 20px;">
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon revenue">
            <el-icon><component :is="icons.Wallet" /></el-icon>
          </div>
          <div class="stat-content">
            <div class="stat-value">¥{{ summary.revenue }}</div>
            <div class="stat-label">总收入</div>
            <div class="stat-change positive">↑ 12.5%</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon orders">
            <el-icon><component :is="icons.ShoppingCart" /></el-icon>
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ summary.orders }}</div>
            <div class="stat-label">订单数</div>
            <div class="stat-change positive">↑ 8.3%</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon customers">
            <el-icon><component :is="icons.User" /></el-icon>
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ summary.customers }}</div>
            <div class="stat-label">客户数</div>
            <div class="stat-change negative">↓ 2.1%</div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stat-card">
          <div class="stat-icon conversion">
            <el-icon><component :is="icons.Percentage" /></el-icon>
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ summary.conversion_rate }}%</div>
            <div class="stat-label">转化率</div>
            <div class="stat-change positive">↑ 3.2%</div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20">
      <el-col :span="12">
        <el-card title="销售趋势">
          <div class="chart-container">
            <canvas ref="salesChart"></canvas>
          </div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card title="订单来源分布">
          <div class="chart-container">
            <canvas ref="sourceChart"></canvas>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="8">
        <el-card title="商品销售排行">
          <el-table :data="topProducts" border>
            <el-table-column prop="rank" label="排名">
              <template #default="scope">
                <span class="rank-badge" :class="getRankClass(scope.row.rank)">{{ scope.row.rank }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="name" label="商品名称" />
            <el-table-column prop="sales" label="销量" />
            <el-table-column prop="revenue" label="销售额">
              <template #default="scope">¥{{ scope.row.revenue }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card title="平台销售对比">
          <div class="platform-comparison">
            <div v-for="platform in platformData" :key="platform.name" class="platform-item">
              <div class="platform-header">
                <span class="platform-name">{{ platform.name }}</span>
                <span class="platform-value">¥{{ platform.revenue }}</span>
              </div>
              <div class="platform-bar">
                <div class="platform-fill" :style="{ width: platform.percentage + '%' }" :class="platform.name"></div>
              </div>
              <div class="platform-info">
                <span>{{ platform.orders }} 订单</span>
                <span>{{ platform.percentage }}%</span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card title="实时数据">
          <div class="real-time-data">
            <div class="data-item">
              <div class="data-icon">
                <el-icon><component :is="icons.Activity" /></el-icon>
              </div>
              <div class="data-info">
                <div class="data-value">{{ realtime.active_users }}</div>
                <div class="data-label">在线用户</div>
              </div>
            </div>
            <div class="data-item">
              <div class="data-icon">
                <el-icon><component :is="icons.ShoppingBag" /></el-icon>
              </div>
              <div class="data-info">
                <div class="data-value">{{ realtime.new_orders }}</div>
                <div class="data-label">今日新订单</div>
              </div>
            </div>
            <div class="data-item">
              <div class="data-icon">
                <el-icon><component :is="icons.Clock" /></el-icon>
              </div>
              <div class="data-info">
                <div class="data-value">{{ realtime.avg_order_time }}</div>
                <div class="data-label">平均订单时长</div>
              </div>
            </div>
            <div class="data-item">
              <div class="data-icon">
                <el-icon><component :is="icons.TrendingUp" /></el-icon>
              </div>
              <div class="data-info">
                <div class="data-value">{{ realtime.conversion_rate }}%</div>
                <div class="data-label">实时转化率</div>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import * as icons from '@element-plus/icons-vue'

const stores = ref([
  { id: 1, name: '抖音官方旗舰店' },
  { id: 2, name: '拼多多专营店' },
  { id: 3, name: '淘宝皇冠店' },
  { id: 4, name: '京东自营店' },
  { id: 5, name: '小红书种草店' }
])

const selectedStore = ref('')
const timeRange = ref('month')

const summary = ref({
  revenue: '128,650.00',
  orders: 856,
  customers: 623,
  conversion_rate: 3.8
})

const topProducts = ref([
  { rank: 1, name: '智能蓝牙耳机Pro', sales: 320, revenue: 63968 },
  { rank: 2, name: '纯棉毛巾套装', sales: 1200, revenue: 47880 },
  { rank: 3, name: '家用收纳箱套装', sales: 890, revenue: 53351 },
  { rank: 4, name: '运动休闲T恤', sales: 560, revenue: 50344 },
  { rank: 5, name: '护肤精华液', sales: 280, revenue: 83720 }
])

const platformData = ref([
  { name: '抖音', revenue: 45200, orders: 280, percentage: 35 },
  { name: '淘宝', revenue: 32800, orders: 220, percentage: 25 },
  { name: '拼多多', revenue: 28500, orders: 310, percentage: 22 },
  { name: '京东', revenue: 15800, orders: 80, percentage: 12 },
  { name: '小红书', revenue: 6350, orders: 40, percentage: 5 }
])

const realtime = ref({
  active_users: 156,
  new_orders: 8,
  avg_order_time: '4.5分钟',
  conversion_rate: 4.2
})

const getRankClass = (rank: number): string => {
  if (rank === 1) return 'gold'
  if (rank === 2) return 'silver'
  if (rank === 3) return 'bronze'
  return ''
}

const refreshData = () => {
  realtime.value.active_users = Math.floor(Math.random() * 100) + 100
  realtime.value.new_orders = Math.floor(Math.random() * 10) + 5
}

onMounted(() => {
  refreshData()
})
</script>

<style scoped>
.data {
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

.stat-card {
  display: flex;
  align-items: center;
  padding: 20px;
}

.stat-icon {
  width: 50px;
  height: 50px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  margin-right: 15px;
}

.stat-icon.revenue {
  background-color: #dbeafe;
  color: #3b82f6;
}

.stat-icon.orders {
  background-color: #dcfce7;
  color: #22c55e;
}

.stat-icon.customers {
  background-color: #fef3c7;
  color: #f59e0b;
}

.stat-icon.conversion {
  background-color: #e0e7ff;
  color: #6366f1;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #1f2937;
}

.stat-label {
  font-size: 12px;
  color: #6b7280;
  margin-top: 2px;
}

.stat-change {
  font-size: 12px;
  margin-top: 4px;
}

.stat-change.positive {
  color: #22c55e;
}

.stat-change.negative {
  color: #ef4444;
}

.chart-container {
  height: 250px;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f9fafb;
  border-radius: 8px;
}

.rank-badge {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background-color: #e5e7eb;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  color: #6b7280;
}

.rank-badge.gold {
  background-color: #fcd34d;
  color: #92400e;
}

.rank-badge.silver {
  background-color: #e5e7eb;
  color: #4b5563;
}

.rank-badge.bronze {
  background-color: #fdba74;
  color: #9a3412;
}

.platform-comparison {
  padding: 10px;
}

.platform-item {
  margin-bottom: 15px;
}

.platform-item:last-child {
  margin-bottom: 0;
}

.platform-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 5px;
}

.platform-name {
  font-size: 14px;
  font-weight: 600;
  color: #374151;
}

.platform-value {
  font-size: 14px;
  font-weight: 600;
  color: #3b82f6;
}

.platform-bar {
  height: 8px;
  background-color: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
}

.platform-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
}

.platform-fill.抖音 { background-color: #ff2c55; }
.platform-fill.淘宝 { background-color: #ff4400; }
.platform-fill.拼多多 { background-color: #ff4d4f; }
.platform-fill.京东 { background-color: #ef3e36; }
.platform-fill.小红书 { background-color: #ff6b6b; }

.platform-info {
  display: flex;
  justify-content: space-between;
  margin-top: 5px;
  font-size: 12px;
  color: #6b7280;
}

.real-time-data {
  padding: 10px;
}

.data-item {
  display: flex;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #e5e7eb;
}

.data-item:last-child {
  border-bottom: none;
}

.data-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background-color: #f3f4f6;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  color: #6b7280;
  margin-right: 15px;
}

.data-info {
  flex: 1;
}

.data-value {
  font-size: 18px;
  font-weight: bold;
  color: #1f2937;
}

.data-label {
  font-size: 12px;
  color: #6b7280;
}
</style>
