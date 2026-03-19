from influxdb_client import InfluxDBClient

client = InfluxDBClient(
    url="https://timeseries.hevs.ch",
    token="ixOI8jiwG1nn6a2MaE1pGa8XCiIJ2rqEX6ZCnluhwAyeZcrT6FHoDgnQhNy5k0YmVrk7hZGPpvb_5aaA-ZxhIw==",
    org="HESSOVS"
)
query_api = client.query_api()

query = '''
import "influxdata/influxdb/schema"
schema.fieldKeys(bucket: "MeteoSuisse")
'''
result = query_api.query(query)
for table in result:
    for r in table.records:
        print(r.values)
client.close()