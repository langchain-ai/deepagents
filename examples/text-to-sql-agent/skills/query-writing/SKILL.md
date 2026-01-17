---
name: query-writing
description: 用于编写和执行 SQL 查询——从简单单表查询到复杂多表 JOIN 与聚合
---

# SQL 查询编写技能

## 何时使用此技能

当你需要通过编写并执行 SQL 查询来回答问题时，使用此技能。

## 简单查询流程

针对单表的直观问题：

1. **确定表** - 哪个表包含数据？
2. **获取表结构** - 使用 `sql_db_schema` 查看列
3. **编写查询** - SELECT 相关列并配合 WHERE/LIMIT/ORDER BY
4. **执行** - 使用 `sql_db_query` 运行
5. **整理答案** - 清晰呈现结果

## 复杂查询流程

针对需要多表联查的问题：

### 1. 制定方案
**使用 `write_todos` 拆解任务：**
- 识别所需的全部表
- 映射关系（外键）
- 规划 JOIN 结构
- 确定聚合方式

### 2. 查看表结构
对每个表使用 `sql_db_schema`，找出联结列与所需字段。

### 3. 构造查询
- SELECT - 列与聚合
- FROM/JOIN - 通过 FK = PK 连接表
- WHERE - 聚合前过滤
- GROUP BY - 所有非聚合列
- ORDER BY - 有意义地排序
- LIMIT - 默认 5 行

### 4. 校验并执行
检查所有 JOIN 都有条件、GROUP BY 正确，然后执行查询。

## 示例：按国家统计收入
```sql
SELECT
    c.Country,
    ROUND(SUM(i.Total), 2) as TotalRevenue
FROM Invoice i
INNER JOIN Customer c ON i.CustomerId = c.CustomerId
GROUP BY c.Country
ORDER BY TotalRevenue DESC
LIMIT 5;
```

## 质量规范

- 只查询相关列（不使用 SELECT *）
- 始终使用 LIMIT（默认 5 行）
- 使用表别名以提升可读性
- 复杂查询：使用 write_todos 进行规划
- 绝不使用 DML 语句（INSERT、UPDATE、DELETE、DROP）
