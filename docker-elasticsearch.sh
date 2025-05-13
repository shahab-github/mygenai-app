docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" \
    -e "xpack.security.transport.ssl.enabled=false" -e "xpack.security.http.ssl.enabled=false" \
    --name elasticsearch elasticsearch:8.12.1


docker run -d \
  --name kibana \
  -p 5601:5601 \
  -e ELASTICSEARCH_HOSTS=http://elasticsearch:9200 \
  --link elasticsearch:elasticsearch \
  kibana:8.12.1
