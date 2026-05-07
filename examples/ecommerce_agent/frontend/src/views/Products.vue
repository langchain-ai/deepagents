<template>
  <div class="products">
    <div class="toolbar">
      <el-select v-model="filterStore" placeholder="选择店铺">
        <el-option label="全部店铺" value="" />
        <el-option v-for="store in stores" :key="store.id" :label="store.name" :value="store.id" />
      </el-select>
      <el-select v-model="filterStatus" placeholder="商品状态">
        <el-option label="全部状态" value="" />
        <el-option label="上架中" value="online" />
        <el-option label="已下架" value="offline" />
        <el-option label="审核中" value="reviewing" />
        <el-option label="违规下架" value="violation" />
      </el-select>
      <el-select v-model="filterCategory" placeholder="商品分类">
        <el-option label="全部分类" value="" />
        <el-option label="数码产品" value="数码产品" />
        <el-option label="服装鞋帽" value="服装鞋帽" />
        <el-option label="家居用品" value="家居用品" />
        <el-option label="食品饮料" value="食品饮料" />
        <el-option label="美妆护肤" value="美妆护肤" />
      </el-select>
      <el-button type="primary" @click="showAddModal = true" icon="Plus">添加商品</el-button>
      <el-button @click="exportProducts" icon="Download">导出商品</el-button>
    </div>

    <el-row :gutter="20" style="margin-bottom: 20px;">
      <el-col :span="4">
        <el-card class="stat-card">
          <div class="stat-value">{{ stats.total }}</div>
          <div class="stat-label">商品总数</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card success">
          <div class="stat-value">{{ stats.online }}</div>
          <div class="stat-label">上架中</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card warning">
          <div class="stat-value">{{ stats.offline }}</div>
          <div class="stat-label">已下架</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card danger">
          <div class="stat-value">{{ stats.low_stock }}</div>
          <div class="stat-label">库存不足</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card info">
          <div class="stat-value">{{ stats.total_sales }}</div>
          <div class="stat-label">总销量</div>
        </el-card>
      </el-col>
      <el-col :span="4">
        <el-card class="stat-card primary">
          <div class="stat-value">¥{{ stats.avg_price }}</div>
          <div class="stat-label">平均价格</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card class="products-table-card">
      <el-table :data="filteredProducts" border>
        <el-table-column prop="product_id" label="商品ID" />
        <el-table-column prop="title" label="商品标题" />
        <el-table-column prop="category" label="分类" />
        <el-table-column prop="price" label="价格">
          <template #default="scope">¥{{ scope.row.price }}</template>
        </el-table-column>
        <el-table-column prop="original_price" label="原价">
          <template #default="scope">¥{{ scope.row.original_price }}</template>
        </el-table-column>
        <el-table-column prop="stock" label="库存">
          <template #default="scope">
            <span :class="scope.row.stock < 20 ? 'low-stock' : ''">{{ scope.row.stock }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="sales" label="销量" />
        <el-table-column prop="status" label="状态">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.status)">{{ getStatusName(scope.row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="update_time" label="更新时间" />
        <el-table-column label="操作">
          <template #default="scope">
            <el-button size="small" @click="viewProduct(scope.row)">详情</el-button>
            <el-button 
              v-if="scope.row.status === 'online'" 
              size="small" 
              type="warning"
              @click="offlineProduct(scope.row)"
            >下架</el-button>
            <el-button 
              v-if="scope.row.status === 'offline'" 
              size="small" 
              type="success"
              @click="onlineProduct(scope.row)"
            >上架</el-button>
            <el-button size="small" type="danger" @click="deleteProduct(scope.row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog title="商品详情" :visible.sync="showDetailModal">
      <el-form :model="selectedProduct" label-width="100px" disabled>
        <el-form-item label="商品ID">
          <el-input v-model="selectedProduct.product_id" />
        </el-form-item>
        <el-form-item label="商品标题">
          <el-input v-model="selectedProduct.title" />
        </el-form-item>
        <el-form-item label="分类">
          <el-input v-model="selectedProduct.category" />
        </el-form-item>
        <el-form-item label="价格">
          <el-input :value="`¥${selectedProduct.price}`" />
        </el-form-item>
        <el-form-item label="原价">
          <el-input :value="`¥${selectedProduct.original_price}`" />
        </el-form-item>
        <el-form-item label="库存">
          <el-input :value="selectedProduct.stock" />
        </el-form-item>
        <el-form-item label="销量">
          <el-input :value="selectedProduct.sales" />
        </el-form-item>
        <el-form-item label="状态">
          <el-tag :type="getStatusType(selectedProduct.status)">{{ getStatusName(selectedProduct.status) }}</el-tag>
        </el-form-item>
        <el-form-item label="创建时间">
          <el-input v-model="selectedProduct.create_time" />
        </el-form-item>
        <el-form-item label="更新时间">
          <el-input v-model="selectedProduct.update_time" />
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showDetailModal = false">关闭</el-button>
      </div>
    </el-dialog>

    <el-dialog title="添加商品" :visible.sync="showAddModal">
      <el-form :model="form" label-width="100px">
        <el-form-item label="商品标题" required>
          <el-input v-model="form.title" />
        </el-form-item>
        <el-form-item label="分类" required>
          <el-select v-model="form.category">
            <el-option label="数码产品" value="数码产品" />
            <el-option label="服装鞋帽" value="服装鞋帽" />
            <el-option label="家居用品" value="家居用品" />
            <el-option label="食品饮料" value="食品饮料" />
            <el-option label="美妆护肤" value="美妆护肤" />
          </el-select>
        </el-form-item>
        <el-form-item label="价格" required>
          <el-input v-model="form.price" type="number" />
        </el-form-item>
        <el-form-item label="原价">
          <el-input v-model="form.original_price" type="number" />
        </el-form-item>
        <el-form-item label="库存">
          <el-input v-model="form.stock" type="number" />
        </el-form-item>
        <el-form-item label="店铺" required>
          <el-select v-model="form.store_id">
            <el-option v-for="store in stores" :key="store.id" :label="store.name" :value="store.id" />
          </el-select>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showAddModal = false">取消</el-button>
        <el-button type="primary" @click="saveProduct">保存</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Product {
  product_id: string
  store_id: number
  store_name: string
  platform: string
  title: string
  category: string
  price: number
  original_price: number
  stock: number
  sales: number
  status: string
  create_time: string
  update_time: string
}

const stores = ref([
  { id: 1, name: '抖音官方旗舰店' },
  { id: 2, name: '拼多多专营店' },
  { id: 3, name: '淘宝皇冠店' },
  { id: 4, name: '京东自营店' },
  { id: 5, name: '小红书种草店' }
])

const products = ref<Product[]>([
  { product_id: 'PRD100001', store_id: 1, store_name: '抖音官方旗舰店', platform: 'douyin', title: '智能蓝牙耳机Pro', category: '数码产品', price: 199.90, original_price: 299.00, stock: 150, sales: 320, status: 'online', create_time: '2024-01-15 10:00', update_time: '2024-03-01 14:00' },
  { product_id: 'PRD100002', store_id: 1, store_name: '抖音官方旗舰店', platform: 'douyin', title: '运动休闲T恤', category: '服装鞋帽', price: 89.90, original_price: 129.00, stock: 8, sales: 560, status: 'online', create_time: '2024-01-20 09:00', update_time: '2024-02-28 16:00' },
  { product_id: 'PRD100003', store_id: 2, store_name: '拼多多专营店', platform: 'pinduoduo', title: '家用收纳箱套装', category: '家居用品', price: 59.90, original_price: 89.00, stock: 200, sales: 890, status: 'online', create_time: '2024-02-01 11:00', update_time: '2024-03-01 10:00' },
  { product_id: 'PRD100004', store_id: 3, store_name: '淘宝皇冠店', platform: 'taobao', title: '进口零食礼盒', category: '食品饮料', price: 128.00, original_price: 168.00, stock: 120, sales: 450, status: 'offline', create_time: '2024-02-10 14:00', update_time: '2024-02-25 09:00' },
  { product_id: 'PRD100005', store_id: 4, store_name: '京东自营店', platform: 'jingdong', title: '护肤精华液', category: '美妆护肤', price: 299.00, original_price: 399.00, stock: 60, sales: 280, status: 'online', create_time: '2024-02-15 10:00', update_time: '2024-03-01 11:00' },
  { product_id: 'PRD100006', store_id: 5, store_name: '小红书种草店', platform: 'xiaohongshu', title: '便携充电宝', category: '数码产品', price: 79.90, original_price: 99.00, stock: 300, sales: 150, status: 'reviewing', create_time: '2024-02-20 15:00', update_time: '2024-02-20 15:00' },
  { product_id: 'PRD100007', store_id: 2, store_name: '拼多多专营店', platform: 'pinduoduo', title: '纯棉毛巾套装', category: '家居用品', price: 39.90, original_price: 59.00, stock: 500, sales: 1200, status: 'online', create_time: '2024-01-25 09:00', update_time: '2024-03-01 08:00' },
  { product_id: 'PRD100008', store_id: 3, store_name: '淘宝皇冠店', platform: 'taobao', title: '男士休闲鞋', category: '服装鞋帽', price: 199.00, original_price: 299.00, stock: 15, sales: 780, status: 'violation', create_time: '2024-01-30 14:00', update_time: '2024-02-10 16:00' }
])

const filterStore = ref('')
const filterStatus = ref('')
const filterCategory = ref('')
const showDetailModal = ref(false)
const showAddModal = ref(false)

const selectedProduct = ref<Product>({
  product_id: '', store_id: 0, store_name: '', platform: '', title: '', category: '',
  price: 0, original_price: 0, stock: 0, sales: 0, status: '', create_time: '', update_time: ''
})

const form = ref({
  title: '',
  category: '数码产品',
  price: 0,
  original_price: 0,
  stock: 100,
  store_id: 1
})

const stats = computed(() => {
  const filtered = filteredProducts.value
  return {
    total: filtered.length,
    online: filtered.filter(p => p.status === 'online').length,
    offline: filtered.filter(p => p.status === 'offline').length,
    low_stock: filtered.filter(p => p.stock < 20).length,
    total_sales: filtered.reduce((sum, p) => sum + p.sales, 0),
    avg_price: filtered.length > 0 ? (filtered.reduce((sum, p) => sum + p.price, 0) / filtered.length).toFixed(2) : '0.00'
  }
})

const filteredProducts = computed(() => {
  return products.value.filter(product => {
    if (filterStore.value && product.store_id !== parseInt(filterStore.value)) return false
    if (filterStatus.value && product.status !== filterStatus.value) return false
    if (filterCategory.value && product.category !== filterCategory.value) return false
    return true
  })
})

const getStatusName = (status: string): string => {
  const names: Record<string, string> = {
    online: '上架中',
    offline: '已下架',
    reviewing: '审核中',
    violation: '违规下架'
  }
  return names[status] || status
}

const getStatusType = (status: string): string => {
  const types: Record<string, string> = {
    online: 'success',
    offline: 'warning',
    reviewing: 'info',
    violation: 'danger'
  }
  return types[status] || 'default'
}

const viewProduct = (product: Product) => {
  selectedProduct.value = product
  showDetailModal.value = true
}

const onlineProduct = (product: Product) => {
  product.status = 'online'
}

const offlineProduct = (product: Product) => {
  product.status = 'offline'
}

const deleteProduct = (product: Product) => {
  if (confirm(`确定删除商品 "${product.title}" 吗？`)) {
    products.value = products.value.filter(p => p.product_id !== product.product_id)
  }
}

const saveProduct = () => {
  const store = stores.value.find(s => s.id === form.value.store_id)
  products.value.push({
    product_id: `PRD${Date.now()}`,
    store_id: form.value.store_id,
    store_name: store?.name || '',
    platform: 'douyin',
    title: form.value.title,
    category: form.value.category,
    price: form.value.price,
    original_price: form.value.original_price || form.value.price,
    stock: form.value.stock,
    sales: 0,
    status: 'online',
    create_time: new Date().toLocaleString('zh-CN'),
    update_time: new Date().toLocaleString('zh-CN')
  })
  showAddModal.value = false
  form.value = { title: '', category: '数码产品', price: 0, original_price: 0, stock: 100, store_id: 1 }
}

const exportProducts = () => {
  const headers = ['商品ID', '店铺', '标题', '分类', '价格', '原价', '库存', '销量', '状态', '更新时间']
  const rows = filteredProducts.value.map(product => [
    product.product_id,
    product.store_name,
    product.title,
    product.category,
    product.price,
    product.original_price,
    product.stock,
    product.sales,
    getStatusName(product.status),
    product.update_time
  ])
  
  const csv = [headers.join(','), ...rows.map(row => row.join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `products_${new Date().toISOString().split('T')[0]}.csv`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}
</script>

<style scoped>
.products {
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

.low-stock {
  color: #ef4444;
  font-weight: bold;
}

.products-table-card {
  margin-top: 20px;
}
</style>
