<template>
  <el-container class="app-container">
    <el-aside width="200px" class="sidebar">
      <div class="logo">
        <h2>电商助手</h2>
      </div>
      <el-menu :default-active="activeMenu" class="sidebar-menu" router>
        <el-menu-item index="/">
          <el-icon><component :is="icons.Dashboard" /></el-icon>
          <span>控制台</span>
        </el-menu-item>
        <el-menu-item index="/stores">
          <el-icon><component :is="icons.Store" /></el-icon>
          <span>店铺管理</span>
        </el-menu-item>
        <el-menu-item index="/tasks">
          <el-icon><component :is="icons.Task" /></el-icon>
          <span>任务管理</span>
        </el-menu-item>
        <el-menu-item index="/scheduled-tasks">
          <el-icon><component :is="icons.Clock" /></el-icon>
          <span>定时任务</span>
        </el-menu-item>
        <el-menu-item index="/elements">
          <el-icon><component :is="icons.Code" /></el-icon>
          <span>元素管理</span>
        </el-menu-item>
        <el-menu-item index="/orders">
          <el-icon><component :is="icons.ShoppingCart" /></el-icon>
          <span>订单管理</span>
        </el-menu-item>
        <el-menu-item index="/products">
          <el-icon><component :is="icons.Package" /></el-icon>
          <span>商品管理</span>
        </el-menu-item>
        <el-menu-item index="/data">
          <el-icon><component :is="icons.BarChart" /></el-icon>
          <span>数据分析</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="header">
        <div class="header-content">
          <h1>{{ pageTitle }}</h1>
          <div class="header-right">
            <el-button @click="refreshData" icon="Refresh" size="small">刷新</el-button>
          </div>
        </div>
      </el-header>
      <el-main class="main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import * as icons from '@element-plus/icons-vue'

const route = useRoute()

const activeMenu = computed(() => route.path)

const pageTitleMap: Record<string, string> = {
  '/': '控制台',
  '/stores': '店铺管理',
  '/tasks': '任务管理',
  '/workflow': '工作流',
  '/elements': 'DOM元素管理',
  '/scheduled-tasks': '定时任务',
  '/data': '数据分析',
  '/orders': '订单管理',
  '/products': '商品管理'
}

const pageTitle = computed(() => pageTitleMap[route.path] || '控制台')

const refreshData = () => {
  // 刷新数据逻辑
}
</script>

<style scoped>
.app-container {
  height: 100vh;
}

.sidebar {
  background-color: #1f2937;
  color: white;
}

.logo {
  padding: 20px;
  text-align: center;
  border-bottom: 1px solid #374151;
}

.logo h2 {
  margin: 0;
  font-size: 18px;
}

.sidebar-menu {
  border-right: none;
  height: calc(100% - 60px);
}

.sidebar-menu :deep(.el-menu-item) {
  color: #9ca3af;
}

.sidebar-menu :deep(.el-menu-item.is-active) {
  color: #3b82f6;
  background-color: rgba(59, 130, 246, 0.1);
}

.header {
  background-color: white;
  border-bottom: 1px solid #e5e7eb;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
}

.header h1 {
  margin: 0;
  font-size: 18px;
  color: #1f2937;
}

.header-right {
  display: flex;
  gap: 10px;
}

.main {
  padding: 20px;
  overflow-y: auto;
}
</style>
