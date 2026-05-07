<template>
  <div class="orders">
    <div class="toolbar">
      <el-select v-model="filterStore" placeholder="选择店铺">
        <el-option label="全部店铺" value="" />
        <el-option v-for="store in stores" :key="store.id" :label="store.name" :value="store.id" />
      </el-select>
      <el-select v-model="filterStatus" placeholder="订单状态">
        <el-option label="全部状态" value="" />
        <el-option label="待付款" value="pending_pay" />
        <el-option label="待发货" value="pending_ship" />
        <el-option label="已发货" value="shipped" />
        <el-option label="待收货" value="pending_receive" />
        <el-option label="已完成" value="completed" />
        <el-option label="已取消" value="cancelled" />
      </el-select>
      <el-date-picker v-model="filterDate" type="date" placeholder="选择日期" />
      <el-button type="primary" @click="exportOrders" icon="Download">导出订单</el-button>
    </div>

    <el-row :gutter="20" style="margin-bottom: 20px;">
      <el-col :span="4">
        <el-card class="stat-card">
          <div class="stat-value">{{ stats.total }}</div>
          <div class="stat-label">订单总数</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card warning">
          <div class="stat-value">{{ stats.pending_ship }}</div>
          <div class="stat-label">待发货</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card success">
          <div class="stat-value">¥{{ stats.total_amount }}</div>
          <div class="stat-label">订单总金额</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card danger">
          <div class="stat-value">{{ stats.cancelled }}</div>
          <div class="stat-label">已取消</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card info">
          <div class="stat-value">{{ stats.completed }}</div>
          <div class="stat-label">已完成</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card primary">
          <div class="stat-value">{{ stats.pending_pay }}</div>
          <div class="stat-label">待付款</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card class="orders-table-card">
      <el-table :data="filteredOrders" border>
        <el-table-column prop="order_id" label="订单号" />
        <el-table-column prop="store_name" label="店铺" />
        <el-table-column prop="platform" label="平台">
          <template #default="scope">
            <span class="platform-tag" :class="scope.row.platform">{{ getPlatformName(scope.row.platform) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="buyer_name" label="买家" />
        <el-table-column prop="item_count" label="商品数量" />
        <el-table-column prop="amount" label="金额">
          <template #default="scope">¥{{ scope.row.amount }}</template>
        </el-table-column>
        <el-table-column prop="status" label="状态">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.status)">{{ getStatusName(scope.row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="create_time" label="创建时间" />
        <el-table-column label="操作">
          <template #default="scope">
            <el-button size="small" @click="viewOrder(scope.row)">详情</el-button>
            <el-button 
              v-if="scope.row.status === 'pending_ship'" 
              size="small" 
              type="primary"
              @click="shipOrder(scope.row)"
            >发货</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog title="订单详情" :visible.sync="showDetailModal">
      <el-form :model="selectedOrder" label-width="100px" disabled>
        <el-form-item label="订单号">
          <el-input v-model="selectedOrder.order_id" />
        </el-form-item>
        <el-form-item label="店铺">
          <el-input v-model="selectedOrder.store_name" />
        </el-form-item>
        <el-form-item label="平台">
          <el-input :value="getPlatformName(selectedOrder.platform)" />
        </el-form-item>
        <el-form-item label="买家">
          <el-input v-model="selectedOrder.buyer_name" />
        </el-form-item>
        <el-form-item label="商品数量">
          <el-input :value="selectedOrder.item_count" />
        </el-form-item>
        <el-form-item label="订单金额">
          <el-input :value="`¥${selectedOrder.amount}`" />
        </el-form-item>
        <el-form-item label="订单状态">
          <el-tag :type="getStatusType(selectedOrder.status)">{{ getStatusName(selectedOrder.status) }}</el-tag>
        </el-form-item>
        <el-form-item label="创建时间">
          <el-input v-model="selectedOrder.create_time" />
        </el-form-item>
        <el-form-item label="付款时间">
          <el-input v-model="selectedOrder.pay_time" />
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showDetailModal = false">关闭</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Order {
  order_id: string
  store_id: number
  store_name: string
  platform: string
  status: string
  amount: number
  item_count: number
  buyer_name: string
  create_time: string
  pay_time: string
}

const stores = ref([
  { id: 1, name: '抖音官方旗舰店' },
  { id: 2, name: '拼多多专营店' },
  { id: 3, name: '淘宝皇冠店' },
  { id: 4, name: '京东自营店' },
  { id: 5, name: '小红书种草店' }
])

const orders = ref<Order[]>([
  { order_id: 'ORD100001', store_id: 1, store_name: '抖音官方旗舰店', platform: 'douyin', status: 'pending_ship', amount: 129.90, item_count: 2, buyer_name: '用户A', create_time: '2024-03-01 14:30:00', pay_time: '2024-03-01 14:35:00' },
  { order_id: 'ORD100002', store_id: 2, store_name: '拼多多专营店', platform: 'pinduoduo', status: 'shipped', amount: 59.90, item_count: 1, buyer_name: '用户B', create_time: '2024-03-01 13:20:00', pay_time: '2024-03-01 13:25:00' },
  { order_id: 'ORD100003', store_id: 3, store_name: '淘宝皇冠店', platform: 'taobao', status: 'completed', amount: 299.00, item_count: 3, buyer_name: '用户C', create_time: '2024-02-28 16:45:00', pay_time: '2024-02-28 16:50:00' },
  { order_id: 'ORD100004', store_id: 4, store_name: '京东自营店', platform: 'jingdong', status: 'pending_receive', amount: 199.90, item_count: 1, buyer_name: '用户D', create_time: '2024-02-28 10:15:00', pay_time: '2024-02-28 10:20:00' },
  { order_id: 'ORD100005', store_id: 1, store_name: '抖音官方旗舰店', platform: 'douyin', status: 'pending_pay', amount: 89.00, item_count: 1, buyer_name: '用户E', create_time: '2024-03-01 15:00:00', pay_time: '' },
  { order_id: 'ORD100006', store_id: 5, store_name: '小红书种草店', platform: 'xiaohongshu', status: 'cancelled', amount: 159.00, item_count: 2, buyer_name: '用户F', create_time: '2024-02-27 09:30:00', pay_time: '2024-02-27 09:35:00' },
  { order_id: 'ORD100007', store_id: 2, store_name: '拼多多专营店', platform: 'pinduoduo', status: 'pending_ship', amount: 45.50, item_count: 1, buyer_name: '用户G', create_time: '2024-03-01 11:00:00', pay_time: '2024-03-01 11:05:00' },
  { order_id: 'ORD100008', store_id: 3, store_name: '淘宝皇冠店', platform: 'taobao', status: 'completed', amount: 399.00, item_count: 1, buyer_name: '用户H', create_time: '2024-02-26 14:20:00', pay_time: '2024-02-26 14:25:00' }
])

const filterStore = ref('')
const filterStatus = ref('')
const filterDate = ref('')
const showDetailModal = ref(false)
const selectedOrder = ref<Order>({
  order_id: '', store_id: 0, store_name: '', platform: '', status: '', 
  amount: 0, item_count: 0, buyer_name: '', create_time: '', pay_time: ''
})

const stats = computed(() => {
  const filtered = filteredOrders.value
  return {
    total: filtered.length,
    total_amount: filtered.reduce((sum, o) => sum + o.amount, 0).toFixed(2),
    pending_pay: filtered.filter(o => o.status === 'pending_pay').length,
    pending_ship: filtered.filter(o => o.status === 'pending_ship').length,
    shipped: filtered.filter(o => o.status === 'shipped').length,
    pending_receive: filtered.filter(o => o.status === 'pending_receive').length,
    completed: filtered.filter(o => o.status === 'completed').length,
    cancelled: filtered.filter(o => o.status === 'cancelled').length
  }
})

const filteredOrders = computed(() => {
  return orders.value.filter(order => {
    if (filterStore.value && order.store_id !== parseInt(filterStore.value)) return false
    if (filterStatus.value && order.status !== filterStatus.value) return false
    return true
  })
})

const getPlatformName = (platform: string): string => {
  const names: Record<string, string> = {
    douyin: '抖音',
    pinduoduo: '拼多多',
    taobao: '淘宝',
    jingdong: '京东',
    xiaohongshu: '小红书'
  }
  return names[platform] || platform
}

const getStatusName = (status: string): string => {
  const names: Record<string, string> = {
    pending_pay: '待付款',
    pending_ship: '待发货',
    shipped: '已发货',
    pending_receive: '待收货',
    completed: '已完成',
    cancelled: '已取消'
  }
  return names[status] || status
}

const getStatusType = (status: string): string => {
  const types: Record<string, string> = {
    pending_pay: 'warning',
    pending_ship: 'danger',
    shipped: 'primary',
    pending_receive: 'info',
    completed: 'success',
    cancelled: 'default'
  }
  return types[status] || 'default'
}

const viewOrder = (order: Order) => {
  selectedOrder.value = order
  showDetailModal.value = true
}

const shipOrder = (order: Order) => {
  order.status = 'shipped'
}

const exportOrders = () => {
  const headers = ['订单号', '店铺', '平台', '买家', '商品数量', '金额', '状态', '创建时间', '付款时间']
  const rows = filteredOrders.value.map(order => [
    order.order_id,
    order.store_name,
    getPlatformName(order.platform),
    order.buyer_name,
    order.item_count,
    order.amount,
    getStatusName(order.status),
    order.create_time,
    order.pay_time
  ])
  
  const csv = [headers.join(','), ...rows.map(row => row.join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `orders_${new Date().toISOString().split('T')[0]}.csv`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}
</script>

<style scoped>
.orders {
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
  text-align: center;
  padding: 15px;
}

.stat-card .stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #3b82f6;
}

.stat-card .stat-label {
  font-size: 12px;
  color: #6b7280;
  margin-top: 5px;
}

.stat-card.success .stat-value { color: #22c55e; }
.stat-card.warning .stat-value { color: #f59e0b; }
.stat-card.danger .stat-value { color: #ef4444; }
.stat-card.info .stat-value { color: #06b6d4; }
.stat-card.primary .stat-value { color: #8b5cf6; }

.platform-tag {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
}

.platform-tag.douyin { background-color: #ff2c55; color: white; }
.platform-tag.pinduoduo { background-color: #ff4d4f; color: white; }
.platform-tag.taobao { background-color: #ff4400; color: white; }
.platform-tag.jingdong { background-color: #ef3e36; color: white; }
.platform-tag.xiaohongshu { background-color: #ff6b6b; color: white; }

.orders-table-card {
  margin-top: 20px;
}
</style>
