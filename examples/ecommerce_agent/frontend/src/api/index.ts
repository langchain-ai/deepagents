const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

// Helper for fetch requests
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`)
  }

  return response.json()
}

// ==================== Stores API ====================
export interface Store {
  id: number
  name: string
  platform: string
  is_active: boolean
  created_at: string
  updated_at?: string
}

export interface StoreCreate {
  name: string
  platform: string
  username?: string
  password?: string
  is_active?: boolean
}

export const storesApi = {
  async getAll(): Promise<Store[]> {
    return apiRequest('/stores')
  },
  async getOne(id: number): Promise<Store> {
    return apiRequest(`/stores/${id}`)
  },
  async create(data: StoreCreate): Promise<{ id: number }> {
    return apiRequest('/stores', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },
  async delete(id: number): Promise<void> {
    return apiRequest(`/stores/${id}`, { method: 'DELETE' })
  },
}

// ==================== Tasks API ====================
export interface Task {
  id: number
  store_id: number
  name: string
  task_type: string
  status: string
  progress: number
  current_step?: string
  created_at: string
  started_at?: string
  completed_at?: string
  result?: any
  error_message?: string
}

export interface TaskCreate {
  store_id: number
  task_type: string
  name?: string
}

export const tasksApi = {
  async getAll(storeId?: number): Promise<Task[]> {
    const params = storeId ? `?store_id=${storeId}` : ''
    return apiRequest(`/tasks${params}`)
  },
  async getOne(id: number): Promise<Task> {
    return apiRequest(`/tasks/${id}`)
  },
  async create(data: TaskCreate): Promise<{ id: number }> {
    return apiRequest('/tasks', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },
  async pause(id: number): Promise<void> {
    return apiRequest(`/tasks/${id}/pause`, { method: 'PUT' })
  },
  async resume(id: number): Promise<void> {
    return apiRequest(`/tasks/${id}/resume`, { method: 'PUT' })
  },
  async delete(id: number): Promise<void> {
    return apiRequest(`/tasks/${id}`, { method: 'DELETE' })
  },
}

// ==================== DOM Elements API ====================
export interface DOMElement {
  id: number
  name: string
  platform: string
  page: string
  selector: string
  selector_type: string
  description?: string
  version: number
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface DOMElementCreate {
  platform: string
  page: string
  name: string
  selector: string
  selector_type?: string
  description?: string
  version?: number
  is_active?: boolean
}

export interface DOMElementUpdate {
  name?: string
  selector?: string
  selector_type?: string
  description?: string
  version?: number
  is_active?: boolean
}

export const domElementsApi = {
  async getAll(platform?: string, page?: string): Promise<DOMElement[]> {
    let params = ''
    if (platform) params += `?platform=${platform}`
    if (page) params += `${params ? '&' : '?'}page=${page}`
    return apiRequest(`/dom-elements${params}`)
  },
  async getOne(id: number): Promise<DOMElement> {
    return apiRequest(`/dom-elements/${id}`)
  },
  async create(data: DOMElementCreate): Promise<{ id: number }> {
    return apiRequest('/dom-elements', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },
  async update(id: number, data: DOMElementUpdate): Promise<{ id: number }> {
    return apiRequest(`/dom-elements/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    })
  },
  async delete(id: number): Promise<void> {
    return apiRequest(`/dom-elements/${id}`, { method: 'DELETE' })
  },
}

// ==================== Scheduled Tasks API ====================
export interface ScheduledTask {
  id: number
  store_id: number
  name: string
  task_type: string
  cron_expression: string
  is_active: boolean
  last_run_at?: string
  next_run_at?: string
  created_at: string
}

export interface ScheduledTaskCreate {
  store_id: number
  task_type: string
  name?: string
  cron_expression: string
  is_active?: boolean
}

export const scheduledTasksApi = {
  async getAll(storeId?: number): Promise<ScheduledTask[]> {
    const params = storeId ? `?store_id=${storeId}` : ''
    return apiRequest(`/scheduled-tasks${params}`)
  },
  async create(data: ScheduledTaskCreate): Promise<{ id: number }> {
    return apiRequest('/scheduled-tasks', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },
  async pause(id: number): Promise<void> {
    return apiRequest(`/scheduled-tasks/${id}/pause`, { method: 'PUT' })
  },
  async resume(id: number): Promise<void> {
    return apiRequest(`/scheduled-tasks/${id}/resume`, { method: 'PUT' })
  },
  async delete(id: number): Promise<void> {
    return apiRequest(`/scheduled-tasks/${id}`, { method: 'DELETE' })
  },
}

// ==================== Orders API ====================
export interface Order {
  id: number
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

export const ordersApi = {
  async getAll(storeId?: number): Promise<Order[]> {
    const params = storeId ? `?store_id=${storeId}` : ''
    return apiRequest(`/orders${params}`)
  },
  async getOne(id: number): Promise<Order> {
    return apiRequest(`/orders/${id}`)
  },
}

// ==================== Products API ====================
export interface Product {
  id: number
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

export const productsApi = {
  async getAll(storeId?: number): Promise<Product[]> {
    const params = storeId ? `?store_id=${storeId}` : ''
    return apiRequest(`/products${params}`)
  },
  async getOne(id: number): Promise<Product> {
    return apiRequest(`/products/${id}`)
  },
}

// ==================== Analytics API ====================
export interface AnalyticsSummary {
  revenue: number
  orders: number
  customers: number
  conversion_rate: number
}

export interface TrendData {
  dates: string[]
  values: number[]
}

export interface PlatformStat {
  name: string
  revenue: number
  orders: number
  percentage: number
}

export const analyticsApi = {
  async getSummary(): Promise<AnalyticsSummary> {
    return apiRequest('/analytics/summary')
  },
  async getTrends(): Promise<TrendData> {
    return apiRequest('/analytics/trends')
  },
  async getTopProducts(): Promise<any[]> {
    return apiRequest('/analytics/top-products')
  },
  async getPlatformStats(): Promise<PlatformStat[]> {
    return apiRequest('/analytics/platform-stats')
  },
}
