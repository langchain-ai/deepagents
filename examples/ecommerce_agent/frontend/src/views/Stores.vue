<template>
  <div class="stores">
    <el-card>
      <h2>店铺管理</h2>
      <el-button type="primary" @click="showAddDialog = true" style="margin-bottom: 20px;">
        添加店铺
      </el-button>
      <el-table :data="stores" style="width: 100%">
        <el-table-column prop="id" label="ID" width="80" />
        <el-table-column prop="name" label="店铺名称" width="200" />
        <el-table-column prop="platform" label="平台" width="120">
          <template #default="{ row }">
            <el-tag>{{ row.platform }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="is_active" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="row.is_active ? 'success' : 'danger'">
              {{ row.is_active ? '启用' : '禁用' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" />
        <el-table-column label="操作" width="200">
          <template #default="{ row }">
            <el-button size="small">编辑</el-button>
            <el-button size="small" type="danger">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="showAddDialog" title="添加店铺" width="500">
      <el-form :model="newStore" label-width="100px">
        <el-form-item label="店铺名称">
          <el-input v-model="newStore.name" />
        </el-form-item>
        <el-form-item label="平台">
          <el-select v-model="newStore.platform" placeholder="请选择平台">
            <el-option label="抖音" value="douyin" />
            <el-option label="拼多多" value="pinduoduo" />
            <el-option label="淘宝" value="taobao" />
          </el-select>
        </el-form-item>
        <el-form-item label="账号">
          <el-input v-model="newStore.username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="newStore.password" type="password" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showAddDialog = false">取消</el-button>
        <el-button type="primary" @click="addStore">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'

const stores = ref<any[]>([])
const showAddDialog = ref(false)
const newStore = ref({
  name: '',
  platform: '',
  username: '',
  password: ''
})

const loadStores = async () => {
  try {
    const res = await axios.get('/api/stores')
    stores.value = res.data
  } catch (error) {
    console.error('Failed to load stores:', error)
  }
}

const addStore = async () => {
  try {
    await axios.post('/api/stores', null, {
      params: newStore.value
    })
    showAddDialog.value = false
    await loadStores()
  } catch (error) {
    console.error('Failed to add store:', error)
  }
}

onMounted(() => {
  loadStores()
})
</script>
