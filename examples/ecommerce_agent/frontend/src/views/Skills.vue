<template>
  <div class="skills-view">
    <div class="page-header">
      <h2>🛠️ 技能管理</h2>
      <p class="subtitle">管理 AI Agent 的专业技能，支持 Claude 风格的技能系统</p>
    </div>

    <el-row :gutter="20">
      <el-col :span="6">
        <el-card class="stats-card">
          <div class="stats-content">
            <div class="stats-icon" style="background: #409eff;">
              <el-icon><Tools /></el-icon>
            </div>
            <div class="stats-info">
              <div class="stats-value">{{ skills.length }}</div>
              <div class="stats-label">技能总数</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stats-card">
          <div class="stats-content">
            <div class="stats-icon" style="background: #67c23a;">
              <el-icon><CircleCheck /></el-icon>
            </div>
            <div class="stats-info">
              <div class="stats-value">{{ activeSkills.length }}</div>
              <div class="stats-label">已启用</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stats-card">
          <div class="stats-content">
            <div class="stats-icon" style="background: #e6a23c;">
              <el-icon><Collection /></el-icon>
            </div>
            <div class="stats-info">
              <div class="stats-value">{{ categories.length }}</div>
              <div class="stats-label">技能分类</div>
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="6">
        <el-card class="stats-card">
          <div class="stats-content">
            <div class="stats-icon" style="background: #f56c6c;">
              <el-icon><Clock /></el-icon>
            </div>
            <div class="stats-info">
              <div class="stats-value">{{ lastUpdated || 'N/A' }}</div>
              <div class="stats-label">最后更新</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-card class="skills-card">
      <div class="skills-header">
        <div class="header-left">
          <h3>📦 技能列表</h3>
          <el-tag type="info">支持 Claude 技能格式</el-tag>
        </div>
        <div class="header-actions">
          <el-button type="primary" @click="refreshSkills" icon="Refresh">刷新</el-button>
        </div>
      </div>

      <el-tabs v-model="activeCategory" @tab-change="filterByCategory">
        <el-tab-pane label="全部" name="all" />
        <el-tab-pane v-for="cat in categories" :key="cat" :label="cat" :name="cat" />
      </el-tabs>

      <el-table :data="filteredSkills" border stripe>
        <el-table-column prop="name" label="技能名称" width="200">
          <template #default="scope">
            <div class="skill-name">
              <el-icon><Folder /></el-icon>
              <strong>{{ scope.row.name }}</strong>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" min-width="300">
          <template #default="scope">
            <div class="skill-description">{{ scope.row.description }}</div>
          </template>
        </el-table-column>
        <el-table-column prop="category" label="分类" width="150">
          <template #default="scope">
            <el-tag :type="getCategoryTagType(scope.row.category)">
              {{ scope.row.category }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="platform" label="支持平台" width="200">
          <template #default="scope">
            <div class="platform-tags">
              <el-tag v-for="p in scope.row.platforms" :key="p" size="small" type="info">
                {{ getPlatformName(p) }}
              </el-tag>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="license" label="许可证" width="100" />
        <el-table-column label="操作" width="180" fixed="right">
          <template #default="scope">
            <el-button size="small" type="primary" @click="viewSkillDetail(scope.row)">
              查看详情
            </el-button>
            <el-button size="small" type="success" @click="loadSkillContent(scope.row)">
              加载内容
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      :title="`技能详情 - ${selectedSkill?.name}`"
      :visible.sync="showDetailDialog"
      width="80%"
      top="5vh"
    >
      <div v-if="selectedSkill" class="skill-detail">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="技能名称">
            <strong>{{ selectedSkill.name }}</strong>
          </el-descriptions-item>
          <el-descriptions-item label="分类">
            <el-tag>{{ selectedSkill.category }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="描述" :span="2">
            {{ selectedSkill.description }}
          </el-descriptions-item>
          <el-descriptions-item label="支持平台">
            <el-tag v-for="p in selectedSkill.platforms" :key="p" size="small" type="success" style="margin-right: 5px;">
              {{ getPlatformName(p) }}
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="许可证">
            {{ selectedSkill.license || 'N/A' }}
          </el-descriptions-item>
          <el-descriptions-item label="兼容性">
            {{ selectedSkill.compatibility || 'N/A' }}
          </el-descriptions-item>
          <el-descriptions-item label="最后更新" :span="2">
            {{ selectedSkill.lastUpdated }}
          </el-descriptions-item>
          <el-descriptions-item label="路径" :span="2">
            <code>{{ selectedSkill.path }}</code>
          </el-descriptions-item>
        </el-descriptions>

        <div class="skill-content-section">
          <h4>📄 技能内容</h4>
          <el-card class="skill-content-card">
            <pre class="skill-content">{{ skillContent }}</pre>
          </el-card>
        </div>
      </div>
    </el-dialog>

    <el-dialog
      title="技能使用指南"
      :visible.sync="showGuideDialog"
      width="70%"
    >
      <el-card class="guide-card">
        <h3>🎯 如何使用技能</h3>
        <el-steps :active="currentStep" finish-status="success">
          <el-step title="步骤 1" description="识别任务类型" />
          <el-step title="步骤 2" description="选择合适技能" />
          <el-step title="步骤 3" description="阅读技能文档" />
          <el-step title="步骤 4" description="执行任务" />
        </el-steps>

        <div class="guide-content">
          <h4>📖 技能系统说明</h4>
          <p>技能系统允许 AI Agent 使用专业的、结构化的工作流来完成特定任务。</p>
          
          <h4>🎯 可用技能</h4>
          <ul>
            <li><strong>product-publish</strong>: 在各大电商平台自动发布商品</li>
            <li><strong>good-review</strong>: 管理好评、回复和追评</li>
            <li><strong>data-collection</strong>: 采集订单、销售和推广数据</li>
          </ul>

          <h4>💡 最佳实践</h4>
          <ul>
            <li>根据任务类型选择合适的技能</li>
            <li>阅读技能完整文档了解详细流程</li>
            <li>遵循技能中的操作规范</li>
            <li>使用技能推荐的工具</li>
          </ul>
        </div>

        <div slot="footer">
          <el-button @click="showGuideDialog = false">关闭</el-button>
          <el-button type="primary" @click="nextStep">下一步</el-button>
        </div>
      </el-card>
    </el-dialog>

    <div class="info-card">
      <el-alert
        title="技能系统说明"
        type="info"
        :closable="false"
        show-icon
      >
        <template>
          <p>本项目使用 <strong>DeepAgents</strong> 框架的技能系统，完全兼容 Claude 的技能模式。</p>
          <p>技能文件位于 <code>skills/</code> 目录下，每个技能包含一个 <code>SKILL.md</code> 文件。</p>
          <p>技能系统支持渐进式披露，先显示元数据，需要时再读取完整内容。</p>
        </template>
      </el-alert>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { 
  Tools, 
  CircleCheck, 
  Collection, 
  Clock, 
  Folder,
  Refresh
} from '@element-plus/icons-vue'

export default {
  name: 'Skills',
  components: {
    Tools,
    CircleCheck,
    Collection,
    Clock,
    Folder,
    Refresh
  },
  setup() {
    const skills = ref([])
    const activeCategory = ref('all')
    const showDetailDialog = ref(false)
    const showGuideDialog = ref(false)
    const selectedSkill = ref(null)
    const skillContent = ref('')
    const currentStep = ref(0)

    const categories = computed(() => {
      const cats = [...new Set(skills.value.map(s => s.category))]
      return cats.sort()
    })

    const activeSkills = computed(() => {
      return skills.value
    })

    const filteredSkills = computed(() => {
      if (activeCategory.value === 'all') {
        return skills.value
      }
      return skills.value.filter(s => s.category === activeCategory.value)
    })

    const lastUpdated = computed(() => {
      if (skills.value.length === 0) return 'N/A'
      const dates = skills.value.map(s => s.lastUpdated).filter(d => d)
      if (dates.length === 0) return 'N/A'
      return dates.sort().reverse()[0]
    })

    const loadSkills = async () => {
      try {
        skills.value = [
          {
            name: 'product-publish',
            description: '在各大电商平台自动发布商品，支持商品信息填写、图片上传和提交审核',
            category: 'ecommerce',
            platforms: ['douyin', 'pdd', 'taobao'],
            license: 'MIT',
            compatibility: 'Python 3.10+, Playwright',
            path: '/skills/ecommerce/product-publish/SKILL.md',
            lastUpdated: '2026-05-07'
          },
          {
            name: 'good-review',
            description: '管理电商平台的好评，包括自动回复、追评和好评分析',
            category: 'ecommerce',
            platforms: ['douyin', 'pdd', 'taobao'],
            license: 'MIT',
            compatibility: 'Python 3.10+, Playwright',
            path: '/skills/ecommerce/good-review/SKILL.md',
            lastUpdated: '2026-05-07'
          },
          {
            name: 'data-collection',
            description: '采集电商平台的订单、销售和推广数据，用于分析和报表生成',
            category: 'ecommerce',
            platforms: ['douyin', 'pdd', 'taobao'],
            license: 'MIT',
            compatibility: 'Python 3.10+, Playwright',
            path: '/skills/ecommerce/data-collection/SKILL.md',
            lastUpdated: '2026-05-07'
          }
        ]
        ElMessage.success('技能列表加载成功')
      } catch (error) {
        ElMessage.error('加载技能列表失败')
        console.error(error)
      }
    }

    const refreshSkills = () => {
      loadSkills()
      ElMessage.success('技能列表已刷新')
    }

    const filterByCategory = (category) => {
      activeCategory.value = category
    }

    const viewSkillDetail = (skill) => {
      selectedSkill.value = skill
      skillContent.value = '正在加载技能内容...'
      showDetailDialog.value = true
      loadSkillContent(skill)
    }

    const loadSkillContent = async (skill) => {
      try {
        const contentMap = {
          'product-publish': `# 商品发布技能

## 何时使用
当用户请求发布商品到抖音电商、拼多多、淘宝等电商平台时使用此技能。

## 支持平台
- 抖音电商
- 拼多多
- 淘宝

## 执行步骤

### 1. 前置准备
- 检查是否已登录目标平台
- 如果未登录，先进行登录操作
- 确认当前店铺信息

### 2. 导航到商品发布页
- 根据平台导航到相应的商品发布页面

### 3. 选择商品类目
- 选择正确的商品类目
- 参考历史经验选择适合的类目

### 4. 填写商品信息
- 商品标题：包含关键词，符合平台规范
- 商品描述：详细描述商品特点和规格
- 商品价格：设置合理的销售价格

### 5. 上传商品图片
- 上传商品主图（至少1张）
- 上传商品详情图

### 6. 提交审核
- 检查商品信息完整性
- 提交商品审核
- 等待审核结果`,

          'good-review': `# 好评管理技能

## 何时使用
当用户需要管理电商平台的好评时使用此技能。

## 支持平台
- 抖音电商
- 拼多多
- 淘宝

## 执行步骤

### 1. 导航到评价页面
- 进入"评价管理"或"订单评价"页面
- 筛选未回复的好评

### 2. 查看好评内容
- 阅读用户评价内容
- 了解用户关注点

### 3. 回复好评
- 使用预设回复模板或自定义回复
- 感谢用户的购买和好评

### 4. 记录和分析
- 记录好评关键词
- 分析用户反馈`,

          'data-collection': `# 数据采集技能

## 何时使用
当需要采集电商平台的订单数据、销售数据、推广数据时使用此技能。

## 支持平台
- 抖音电商
- 拼多多
- 淘宝

## 采集数据类型

### 1. 订单数据
- 订单列表
- 订单详情
- 订单状态

### 2. 销售数据
- 商品销量
- 销售额
- 销售趋势

### 3. 推广数据
- 推广效果
- 流量来源
- 转化率

## 执行步骤

### 1. 准备阶段
- 确认采集目标（日期范围）
- 确定数据类型

### 2. 导航到数据页面
- 进入数据中心/数据分析页面
- 设置日期筛选条件

### 3. 数据采集
- 提取页面数据
- 分页获取全部数据

### 4. 数据存储
- 保存到数据库
- 生成数据报告`
        }

        skillContent.value = contentMap[skill.name] || '技能内容加载中...'
      } catch (error) {
        ElMessage.error('加载技能内容失败')
        console.error(error)
      }
    }

    const getCategoryTagType = (category) => {
      const types = {
        'ecommerce': 'success',
        'base': 'primary',
        'advanced': 'warning'
      }
      return types[category] || 'info'
    }

    const getPlatformName = (platform) => {
      const names = {
        'douyin': '抖音',
        'pdd': '拼多多',
        'taobao': '淘宝',
        'jd': '京东',
        'xhs': '小红书'
      }
      return names[platform] || platform
    }

    const nextStep = () => {
      if (currentStep.value < 3) {
        currentStep.value++
      } else {
        showGuideDialog.value = false
        currentStep.value = 0
      }
    }

    onMounted(() => {
      loadSkills()
      setTimeout(() => {
        showGuideDialog.value = true
      }, 1000)
    })

    return {
      skills,
      activeCategory,
      categories,
      activeSkills,
      filteredSkills,
      lastUpdated,
      showDetailDialog,
      showGuideDialog,
      selectedSkill,
      skillContent,
      currentStep,
      refreshSkills,
      filterByCategory,
      viewSkillDetail,
      loadSkillContent,
      getCategoryTagType,
      getPlatformName,
      nextStep
    }
  }
}
</script>

<style scoped>
.skills-view {
  padding: 20px;
}

.page-header {
  margin-bottom: 30px;
}

.page-header h2 {
  margin: 0 0 10px 0;
  font-size: 28px;
  color: #303133;
}

.subtitle {
  color: #909399;
  margin: 0;
}

.stats-card {
  margin-bottom: 20px;
}

.stats-content {
  display: flex;
  align-items: center;
  gap: 15px;
}

.stats-icon {
  width: 60px;
  height: 60px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 28px;
}

.stats-info {
  flex: 1;
}

.stats-value {
  font-size: 32px;
  font-weight: bold;
  color: #303133;
  line-height: 1;
}

.stats-label {
  color: #909399;
  font-size: 14px;
  margin-top: 5px;
}

.skills-card {
  margin: 20px 0;
}

.skills-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 15px;
}

.header-left h3 {
  margin: 0;
  font-size: 20px;
}

.skill-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.skill-description {
  color: #606266;
  line-height: 1.5;
}

.platform-tags {
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
}

.skill-detail {
  padding: 10px;
}

.skill-content-section {
  margin-top: 20px;
}

.skill-content-section h4 {
  margin-bottom: 10px;
  font-size: 16px;
  color: #303133;
}

.skill-content-card {
  max-height: 400px;
  overflow-y: auto;
}

.skill-content {
  background: #f5f7fa;
  padding: 15px;
  border-radius: 4px;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.6;
  margin: 0;
}

.guide-card {
  padding: 10px;
}

.guide-content {
  margin-top: 30px;
}

.guide-content h4 {
  margin-top: 20px;
  margin-bottom: 10px;
  color: #303133;
}

.guide-content ul {
  padding-left: 20px;
}

.guide-content li {
  margin: 8px 0;
  color: #606266;
}

.info-card {
  margin-top: 20px;
}

.info-card p {
  margin: 8px 0;
  line-height: 1.8;
}

.info-card code {
  background: #f5f7fa;
  padding: 2px 6px;
  border-radius: 3px;
  color: #409eff;
}
</style>
