---
name: schema-exploration
description: For discovering and understanding database structure, tables, columns, and relationships
---

# Schema Exploration Skill

## When to Use This Skill

Use this skill when you need to:
- Understand the database structure
- Find which tables contain certain types of data
- Discover column names and data types
- Map relationships between tables
- Answer questions like "What tables are available?" or "What columns does the Customer table have?"

## Workflow

### 1. List All Tables
Use `sql_db_list_tables` tool to see all available tables in the database.

This returns the complete list of tables you can query.

### 2. Get Schema for Specific Tables
Use `sql_db_schema` tool with table names to examine:
- **Column names** - What fields are available
- **Data types** - INTEGER, TEXT, DATETIME, etc.
- **Sample data** - 3 example rows to understand content
- **Primary keys** - Unique identifiers for rows
- **Foreign keys** - Relationships to other tables

### 3. Map Relationships
Identify how tables connect:
- Look for columns ending in "Id" (e.g., CustomerId, ArtistId)
- Foreign keys link to primary keys in other tables
- Document parent-child relationships

### 4. Answer the Question
Provide clear information about:
- Available tables and their purpose
- Column names and what they contain
- How tables relate to each other
- Sample data to illustrate content

## Example: "What tables are available?"

**Step 1:** Use `sql_db_list_tables`

**Response:**
```
The Chinook database has 11 tables:
1. Artist - Music artists
2. Album - Music albums
3. Track - Individual songs
4. Genre - Music genres
5. MediaType - File formats (MP3, AAC, etc.)
6. Playlist - User-created playlists
7. PlaylistTrack - Tracks in playlists
8. Customer - Store customers
9. Employee - Store employees
10. Invoice - Customer purchases
11. InvoiceLine - Individual items in invoices
```

## Example: "What columns does the Customer table have?"

**Step 1:** Use `sql_db_schema` with table name "Customer"

**Response:**
```
The Customer table has these columns:

- CustomerId (INTEGER) - Primary key, unique customer ID
- FirstName (NVARCHAR) - Customer's first name
- LastName (NVARCHAR) - Customer's last name
- Company (NVARCHAR) - Company name (optional)
- Address, City, State, Country, PostalCode - Location fields
- Phone (NVARCHAR) - Phone number
- Fax (NVARCHAR) - Fax number
- Email (NVARCHAR) - Email address
- SupportRepId (INTEGER) - Foreign key to Employee table

Sample data shows customers like:
- Luís Gonçalves from Brazil
- Leonie Köhler from Germany
- François Tremblay from Canada
```

## Example: "How do I find revenue by artist?"

**Step 1:** Identify tables needed
- Artist (has artist names)
- Album (links artists to tracks)
- Track (links albums to sales)
- InvoiceLine (has sales data)
- Invoice (has revenue totals)

**Step 2:** Map relationships
```
Artist (ArtistId)
  ↓ 1:many
Album (ArtistId, AlbumId)
  ↓ 1:many
Track (AlbumId, TrackId)
  ↓ 1:many
InvoiceLine (TrackId, UnitPrice, Quantity)
```

**Response:**
```
To find revenue by artist, you'll need to:
1. Start with the Artist table
2. JOIN to Album on ArtistId
3. JOIN to Track on AlbumId
4. JOIN to InvoiceLine on TrackId
5. SUM the revenue (UnitPrice * Quantity)
6. GROUP BY artist name

This requires the query-writing skill to execute.
```

## Quality Guidelines

**For "list tables" questions:**
- Show all table names
- Add brief descriptions of what each contains
- Group related tables (e.g., music catalog, transactions, people)

**For "describe table" questions:**
- List all columns with data types
- Explain what each column contains
- Show sample data for context
- Note primary and foreign keys
- Explain relationships to other tables

**For "how do I query X" questions:**
- Identify required tables
- Map the JOIN path
- Explain the relationship chain
- Suggest next steps (use query-writing skill)

## Common Exploration Patterns

### Pattern 1: Find a Table
"Which table has customer information?"
→ Use list_tables, then describe Customer table

### Pattern 2: Understand Structure
"What's in the Invoice table?"
→ Use schema tool to show columns and sample data

### Pattern 3: Map Relationships
"How are artists connected to sales?"
→ Trace the foreign key chain: Artist → Album → Track → InvoiceLine → Invoice

## Tips

- Table names in Chinook are singular and capitalized (Customer, not customers)
- Foreign keys typically have "Id" suffix and match a table name
- Use sample data to understand what values look like
- When unsure which table to use, list all tables first
