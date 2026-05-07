<template>
  <div class="elements">
    <div class="toolbar">
      <el-select v-model="selectedPlatform" placeholder="选择平台">
        <el-option v-for="platform in platforms" :key="platform.value" :label="platform.label" :value="platform.value" />
      </el-select>
      <el-select v-model="selectedPage" placeholder="选择页面">
        <el-option v-for="page in pages" :key="page" :label="page" :value="page" />
      </el-select>
      <el-button type="primary" @click="showAddModal = true" icon="Plus">添加元素</el-button>
      <el-button @click="refreshElements" icon="Refresh">刷新</el-button>
    </div>

    <el-card class="elements-card">
      <div class="elements-header">
        <h3>DOM 元素配置</h3>
        <span class="count">共 {{ elements.length }} 个元素</span>
      </div>
      
      <el-table :data="elements" border>
        <el-table-column prop="name" label="元素名称" />
        <el-table-column prop="selector" label="选择器">
          <template #default="scope">
            <code class="selector-code">{{ scope.row.selector }}</code>
          </template>
        </el-table-column>
        <el-table-column prop="selector_type" label="选择器类型">
          <template #default="scope">
            <el-tag :type="getSelectorTypeTag(scope.row.selector_type)">
              {{ getSelectorTypeName(scope.row.selector_type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" />
        <el-table-column prop="version" label="版本" />
        <el-table-column prop="status" label="状态">
          <template #default="scope">
            <el-switch 
              :value="scope.row.status === 'active'" 
              @change="toggleStatus(scope.row)"
              :disabled="scope.row.status === 'deprecated'"
            />
          </template>
        </el-table-column>
        <el-table-column prop="updated_at" label="更新时间" />
        <el-table-column label="操作">
          <template #default="scope">
            <el-button size="small" @click="editElement(scope.row)">编辑</el-button>
            <el-button size="small" type="danger" @click="deleteElement(scope.row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog :title="isEditing ? '编辑元素' : '添加元素'" :visible.sync="showAddModal">
      <el-form :model="form" label-width="100px">
        <el-form-item label="元素名称" required>
          <el-input v-model="form.name" placeholder="例如: username_input" />
        </el-form-item>
        <el-form-item label="选择器" required>
          <el-input v-model="form.selector" placeholder="例如: input[name='username']" />
        </el-form-item>
        <el-form-item label="选择器类型" required>
          <el-select v-model="form.selector_type">
            <el-option label="CSS选择器" value="css" />
            <el-option label="XPath" value="xpath" />
            <el-option label="ID" value="id" />
            <el-option label="Class" value="class" />
            <el-option label="Name" value="name" />
          </el-select>
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="form.description" placeholder="元素描述" />
        </el-form-item>
        <el-form-item label="版本">
          <el-input v-model="form.version" placeholder="1.0.0" />
        </el-form-item>
        <el-form-item label="状态">
          <el-select v-model="form.status">
            <el-option label="启用" value="active" />
            <el-option label="禁用" value="inactive" />
            <el-option label="废弃" value="deprecated" />
          </el-select>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="showAddModal = false">取消</el-button>
        <el-button type="primary" @click="saveElement">保存</el-button>
      </div>
    </el-dialog>

    <el-dialog title="元素详情" :visible.sync="showDetailModal">
      <el-form :model="selectedElement" label-width="100px" disabled>
        <el-form-item label="元素名称">
          <el-input v-model="selectedElement.name" />
        </el-form-item>
        <el-form-item label="选择器">
          <el-input v-model="selectedElement.selector" />
        </el-form-item>
        <el-form-item label="选择器类型">
          <el-input :value="getSelectorTypeName(selectedElement.selector_type)" />
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="selectedElement.description" />
        </el-form-item>
        <el-form-item label="版本">
          <el-input v-model="selectedElement.version" />
        </el-form-item>
        <el-form-item label="平台">
          <el-input :value="getPlatformName(selectedElement.platform)" />
        </el-form-item>
        <el-form-item label="页面">
          <el-input v-model="selectedElement.page" />
        </el-form-item>
        <el-form-item label="创建时间">
          <el-input v-model="selectedElement.created_at" />
        </el-form-item>
        <el-form-item label="更新时间">
          <el-input v-model="selectedElement.updated_at" />
        </el-form-item>
      </el-form>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Element {
  id: number
  name: string
  selector: string
  selector_type: string
  description: string
  version: string
  platform: string
  page: string
  status: string
  created_at: string
  updated_at: string
}

const platforms = [
  { value: 'douyin', label: '抖音' },
  { value: 'pinduoduo', label: '拼多多' },
  { value: 'taobao', label: '淘宝' },
  { value: 'jingdong', label: '京东' },
  { value: 'xiaohongshu', label: '小红书' }
]

const pages = ['login', 'publish', 'reviews', 'orders', 'products']

const selectedPlatform = ref('douyin')
const selectedPage = ref('login')
const showAddModal = ref(false)
const showDetailModal = ref(false)
const isEditing = ref(false)
const selectedElement = ref<Element>({
  id: 0, name: '', selector: '', selector_type: 'css', 
  description: '', version: '1.0.0', platform: 'douyin', 
  page: 'login', status: 'active', created_at: '', updated_at: ''
})

const form = ref({
  id: 0,
  name: '',
  selector: '',
  selector_type: 'css',
  description: '',
  version: '1.0.0',
  status: 'active'
})

const elements = ref<Element[]>([
  { id: 1, name: 'username_input', selector: 'input[name="username"]', selector_type: 'css', description: '用户名输入框', version: '1.0.0', platform: 'douyin', page: 'login', status: 'active', created_at: '2024-01-15 10:00', updated_at: '2024-01-15 10:00' },
  { id: 2, name: 'password_input', selector: 'input[name="password"]', selector_type: 'css', description: '密码输入框', version: '1.0.0', platform: 'douyin', page: 'login', status: 'active', created_at: '2024-01-15 10:05', updated_at: '2024-01-15 10:05' },
  { id: 3, name: 'login_button', selector: '//button[contains(text(),"登录")]', selector_type: 'xpath', description: '登录按钮', version: '1.0.0', platform: 'douyin', page: 'login', status: 'active', created_at: '2024-01-15 10:10', updated_at: '2024-01-15 10:10' },
  { id: 4, name: 'captcha_image', selector: 'captcha-img', selector_type: 'class', description: '验证码图片', version: '1.0.0', platform: 'douyin', page: 'login', status: 'active', created_at: '2024-01-15 10:15', updated_at: '2024-01-15 10:15' },
  { id: 5, name: 'captcha_input', selector: 'captcha', selector_type: 'name', description: '验证码输入框', version: '1.0.0', platform: 'douyin', page: 'login', status: 'inactive', created_at: '2024-01-15 10:20', updated_at: '2024-01-15 10:20' },
  { id: 6, name: 'product_title', selector: '//input[@id="title"]', selector_type: 'xpath', description: '商品标题输入框', version: '1.1.0', platform: 'douyin', page: 'publish', status: 'active', created_at: '2024-01-20 09:00', updated_at: '2024-02-01 14:00' },
  { id: 7, name: 'product_price', selector: '#price', selector_type: 'id', description: '商品价格输入框', version: '1.0.0', platform: 'douyin', page: 'publish', status: 'active', created_at: '2024-01-20 09:05', updated_at: '2024-01-20 09:05' },
  { id: 8, name: 'submit_button', selector: '.submit-btn', selector_type: 'class', description: '提交按钮', version: '1.0.0', platform: 'douyin', page: 'publish', status: 'deprecated', created_at: '2024-01-10 11:00', updated_at: '2024-01-15 16:00' }
])

const getPlatformName = (platform: string): string => {
  const platformMap: Record<string, string> = {
    douyin: '抖音',
    pinduoduo: '拼多多',
    taobao: '淘宝',
    jingdong: '京东',
    xiaohongshu: '小红书'
  }
  return platformMap[platform] || platform
}

const getSelectorTypeName = (type: string): string => {
  const typeMap: Record<string, string> = {
    css: 'CSS选择器',
    xpath: 'XPath',
    id: 'ID',
    class: 'Class',
    name: 'Name'
  }
  return typeMap[type] || type
}

const getSelectorTypeTag = (type: string): string => {
  const typeMap: Record<string, string> = {
    css: 'primary',
    xpath: 'success',
    id: 'warning',
    class: 'info',
    name: 'danger'
  }
  return typeMap[type] || 'info'
}

const refreshElements = () => {}

const toggleStatus = (element: Element) => {
  element.status = element.status === 'active' ? 'inactive' : 'active'
}

const editElement = (element: Element) => {
  isEditing.value = true
  form.value = {
    id: element.id,
    name: element.name,
    selector: element.selector,
    selector_type: element.selector_type,
    description: element.description,
    version: element.version,
    status: element.status
  }
  showAddModal.value = true
}

const deleteElement = (element: Element) => {
  if (confirm(`确定删除元素 "${element.name}" 吗？`)) {
    elements.value = elements.value.filter(e => e.id !== element.id)
  }
}

const saveElement = () => {
  if (isEditing.value) {
    const index = elements.value.findIndex(e => e.id === form.value.id)
    if (index !== -1) {
      elements.value[index] = {
        ...elements.value[index],
        name: form.value.name,
        selector: form.value.selector,
        selector_type: form.value.selector_type,
        description: form.value.description,
        version: form.value.version,
        status: form.value.status,
        updated_at: new Date().toLocaleString('zh-CN')
      }
    }
  } else {
    elements.value.push({
      id: elements.value.length + 1,
      name: form.value.name,
      selector: form.value.selector,
      selector_type: form.value.selector_type,
      description: form.value.description,
      version: form.value.version,
      platform: selectedPlatform.value,
      page: selectedPage.value,
      status: form.value.status,
      created_at: new Date().toLocaleString('zh-CN'),
      updated_at: new Date().toLocaleString('zh-CN')
    })
  }
  showAddModal.value = false
  form.value = { id: 0, name: '', selector: '', selector_type: 'css', description: '', version: '1.0.0', status: 'active' }
  isEditing.value = false
}
</script>

<style scoped>
.elements {
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

.elements-card {
  margin-top: 20px;
}

.elements-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.elements-header h3 {
  margin: 0;
  font-size: 16px;
}

.count {
  font-size: 14px;
  color: #6b7280;
}

.selector-code {
  background-color: #f3f4f6;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  color: #374151;
}
</style>
