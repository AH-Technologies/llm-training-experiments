"""Fix parquet schemas: cast large_string to string for verl compatibility."""
import pyarrow as pa
import pyarrow.parquet as pq

for f in ["attention_based_rewards/data/aime_2025.parquet", "attention_based_rewards/data/amc_2023.parquet"]:
    table = pq.read_table(f)
    for i, field in enumerate(table.schema):
        if field.type == pa.large_string():
            col = table.column(i).cast(pa.string())
            table = table.set_column(i, field.with_type(pa.string()), col)
    pq.write_table(table, f)
    schema = pq.read_schema(f)
    ds_type = schema.field("data_source").type
    ab_type = schema.field("ability").type
    print(f"Fixed {f}: data_source={ds_type}, ability={ab_type}")
