---
name: schema-exploration
description: 用于发现与理解数据库结构、表、列以及表间关系
---

# 数据库结构探索技能

## 何时使用此技能

当你需要：
- 理解数据库结构
- 找到包含某类数据的表
- 发现列名与数据类型
- 梳理表之间的关系
- 回答诸如“有哪些表？”或“Customer 表有哪些列？”之类的问题

## 工作流程

### 1. 列出所有表
使用 `sql_db_list_tables` 工具查看数据库中的全部表。

它会返回你可以查询的完整表清单。

### 2. 获取指定表的结构
使用 `sql_db_schema` 并指定表名，查看：
- **列名** - 可用字段
- **数据类型** - INTEGER、TEXT、DATETIME 等
- **示例数据** - 3 行样例，帮助理解内容
- **主键** - 行的唯一标识
- **外键** - 指向其他表的关系

### 3. 梳理关系
识别表之间的连接方式：
- 查找以 “Id” 结尾的列（如 CustomerId、ArtistId）
- 外键连接到其他表的主键
- 记录父子关系

### 4. 回答问题
清晰提供：
- 可用表及其用途
- 列名及含义
- 表之间的关系
- 用示例数据说明内容

## 示例：“有哪些表？”

**步骤 1：** 使用 `sql_db_list_tables`

**回答：**
```
Chinook 数据库包含 11 张表：
1. Artist - 音乐艺术家
2. Album - 音乐专辑
3. Track - 单曲
4. Genre - 音乐流派
5. MediaType - 文件格式（MP3、AAC 等）
6. Playlist - 用户创建的播放列表
7. PlaylistTrack - 播放列表中的曲目
8. Customer - 商店客户
9. Employee - 商店员工
10. Invoice - 客户购买记录
11. InvoiceLine - 发票中的具体条目
```

## 示例：“Customer 表有哪些列？”

**步骤 1：** 使用 `sql_db_schema` 并指定表名 “Customer”

**回答：**
```
Customer 表包含以下列：

- CustomerId (INTEGER) - 主键，唯一客户 ID
- FirstName (NVARCHAR) - 名
- LastName (NVARCHAR) - 姓
- Company (NVARCHAR) - 公司名（可选）
- Address、City、State、Country、PostalCode - 地址相关字段
- Phone (NVARCHAR) - 电话号码
- Fax (NVARCHAR) - 传真号码
- Email (NVARCHAR) - 电子邮箱
- SupportRepId (INTEGER) - 外键，指向 Employee 表

示例数据包含客户：
- Luís Gonçalves（巴西）
- Leonie Köhler（德国）
- François Tremblay（加拿大）
```

## 示例：“如何按艺术家统计收入？”

**步骤 1：** 识别所需表
- Artist（包含艺术家名）
- Album（连接艺术家与曲目）
- Track（连接专辑与销售）
- InvoiceLine（包含销售数据）
- Invoice（包含收入总额）

**步骤 2：** 梳理关系
```
Artist (ArtistId)
  ↓ 1:many
Album (ArtistId, AlbumId)
  ↓ 1:many
Track (AlbumId, TrackId)
  ↓ 1:many
InvoiceLine (TrackId, UnitPrice, Quantity)
```

**回答：**
```
要按艺术家统计收入，你需要：
1. 从 Artist 表开始
2. 通过 ArtistId JOIN 到 Album
3. 通过 AlbumId JOIN 到 Track
4. 通过 TrackId JOIN 到 InvoiceLine
5. 汇总收入（UnitPrice * Quantity）
6. 按艺术家名称 GROUP BY

这需要使用查询编写技能来执行。
```

## 质量规范

**针对“列出表”的问题：**
- 展示全部表名
- 简要描述每张表的内容
- 按类别归组（如音乐目录、交易、人员）

**针对“描述表”的问题：**
- 列出所有列及其数据类型
- 解释每列包含的内容
- 展示示例数据以提供上下文
- 标注主键与外键
- 说明与其他表的关系

**针对“如何查询 X”的问题：**
- 识别所需表
- 梳理 JOIN 路径
- 解释关系链
- 建议下一步（使用查询编写技能）

## 常见探索模式

### 模式 1：找到表
“哪张表包含客户信息？”
→ 使用 `sql_db_list_tables`，然后描述 Customer 表

### 模式 2：理解结构
“Invoice 表里有什么？”
→ 使用 `sql_db_schema` 展示列与样例数据

### 模式 3：梳理关系
“艺术家和销售是如何关联的？”
→ 追踪外键链：Artist → Album → Track → InvoiceLine → Invoice

## 提示

- Chinook 的表名是单数且首字母大写（Customer，而不是 customers）
- 外键通常以 “Id” 结尾，并匹配表名
- 使用样例数据理解字段取值
- 不确定使用哪张表时，先列出全部表
