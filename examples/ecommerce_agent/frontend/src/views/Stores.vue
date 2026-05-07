<template>
  <div class="stores">
    <div class="toolbar">
      <el-button type="primary" @click="showAddModal = true" icon="Plus">添加店铺</el-button>
      <el-button @click="refreshStores" icon="Refresh">刷新</el-button>
    </div>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="6" v-for="store in stores" :key="store.id">
        <el-card class="store-card" @click="viewStore(store)">
          <div class="store-header">
            <div class="platform-badge" :class="store.platform">
              {{ getPlatformName(store.platform) }}
            </div>
            <el-tag :type="store.status === 'active' ? 'success' : 'warning'">
              {{ store.status === 'active' ? '已启用' : '已禁用' }}
            </el-tag>
          </div>
          <h3 class="store-name">{{ store.name }}</h3>
          <div class="store-info">
            <div>账号: {{ store.account }}</div>
            <div>创建时间: {{ formatDate(store.created_at) }}</div>
          </div>
          <div class="store-actions">
            <el-button size="small" @click.stop="editStore(store)">编辑</el-button>
            <el-button size="small" type="danger" @click.stop="deleteStore(store)">删除</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-dialog title="添加店铺" :visible.sync="showAddModal">
      <el-form :model="form" label-width="80px">
        <el-form-item label="店铺名称">
          <el-input v-model="form.name" />
        </el-form-item>
        <el-form-item label="平台">
          <el-select v-model="form.platform">
            <el-option label="抖音" value="douyin" />
            <el-option label="拼多多" value="pinduoduo" />
            <el-option label="淘宝" value="taobao" />
            <el-option label="京东" value="jingdong" />
            <el-option label="小红书" value="xiaohongshu" />
          </el-select>
        </el-form-item>
        <el-form-item label="账号">
          <el-input v-model="form.account" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="form.password" type="password" />
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showAddModal = false">取消</el-button>
        <el-button type="primary" @click="saveStore">保存</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface Store {
  id: number
  name: string
  platform: string
  account: string
  password: string
  status: string
  created_at: string
}

const stores = ref<Store[]>([
  { id: 1, name: '抖音官方旗舰店', platform: 'douyin', account: 'shop001', password: '******', status: 'active', created_at: '2024-01-15' },
  { id: 2, name: '拼多多专营店', platform: 'pinduoduo', account: 'shop002', password: '******', status: 'active', created_at: '2024-01-20' },
  { id: 3, name: '淘宝皇冠店', platform: 'taobao', account: 'shop003', password: '******', status: 'active', created_at: '2024-02-01' },
  { id: 4, name: '京东自营店', platform: 'jingdong', account: 'shop004', password: '******', status: 'active', created_at: '2024-02-10' },
  { id: 5, name: '小红书种草店', platform: 'xiaohongshu', account: 'shop005', password: '******', status: 'disabled', created_at: '2024-02-15' }
])

const showAddModal = ref(false)
const form = ref({
  name: '',
  platform: 'douyin',
  account: '',
  password: ''
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

const formatDate = (date: string): string => date

const refreshStores = () => {}

const viewStore = (store: Store) => {}

const editStore = (store: Store) => {}

const deleteStore = (store: Store) => {
  if (confirm(`确定删除店铺 "${store.name}" 吗？`)) {
    stores.value = stores.value.filter(s => s.id !== store.id)
  }
}

const saveStore = () => {
  stores.value.push({
    id: stores.value.length + 1,
    name: form.value.name,
    platform: form.value.platform,
    account: form.value.account,
    password: '******',
    status: 'active',
    created_at: new Date().toISOString().split('T')[0]
  })
  showAddModal.value = false
  form.value = { name: '', platform: 'douyin', account: '', password: '' }
}
</script>

<style scoped>
.stores {
  padding: 10px;
}

.toolbar {
  display: flex;
  gap: 10px;
}

.store-card {
  cursor: pointer;
  transition: all 0.3s;
}

.store-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.store-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.platform-badge {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
}

.platform-badge.douyin {
  background-color: #ff2c55;
  color: white;
}

.platform-badge.pinduoduo {
  background-color: #ff4d4f;
  color: white;
}

.platform-badge.taobao {
  background-color: #ff4400;
  color: white;
}

.platform-badge.jingdong {
  background-color: #ef3e36;
  color: white;
}

.platform-badge.xiaohongshu {
  background-color: #ff6b6b;
  color: white;
}

.store-name {
  margin: 0 0 10px 0;
  font-size: 16px;
  color: #1f2937;
}

.store-info {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 15px;
}

.store-info div {
  margin-bottom: 4px;
}

.store-actions {
  display: flex;
  gap: 8px;
}
</style>
