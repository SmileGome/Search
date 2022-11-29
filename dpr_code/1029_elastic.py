import json

##############################################################
# TEST EMBEDDING

origin_file = 'data/clean_anno.json'
emb_file = 'data/result/emb_short/doc_emb_short.txt'

with open(emb_file) as f:
    doc_embedding = f.readlines()

with open(origin_file, "r", encoding="utf-8") as json_file:
    latex_data = json.load(json_file)
    
embedding = doc_embedding[1][1:-2]
embedding = list(map(float, embedding.split(',')))

# embedding 검색 실험
test_embedding = embedding

###############################################################

# embedding 검색 실험
test_embedding = embedding

from elasticsearch import Elasticsearch

es = Elasticsearch(
    cloud_id="MathMatch:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDJmYzA4YmQ1ZTg5YjQ5NGFhODg4YjVmZjE2YWQxODE5JGVlN2U4ZjI4ODQ0ZTRlYzBhYTE2ZjdhNTAwZWQ4YjRk",
    basic_auth=("elastic", "rvUUGH364TvAj6tyS9Y3fitD")
)

body ={
  'size':10, 
  "query": {
    "script_score": {
      "query" : {
        "match_all": {}
        },
      
      "script": 
      {
        "source": "cosineSimilarity(params.query_vector, 'embedding')", 
        "params": 
        {
          "query_vector": test_embedding
        }
      }
    }
    }
    \
}
latex = es.search(index='latex_embedding', body=body)

latex_info = latex['hits']['hits']
for i in range(len(latex_info)):
    id = latex_info[i]['_id']
    info = latex_info[i]['_source']
    doc_id = info['doc_id']
    body = {
		'query': {
				'match': {
						'doc_id': doc_id
                    }
            }
    }
    doc = es.search(index='doc_info', body=body)
    doc_info = doc['hits']['hits'][0]['_source']
    body = {'id': id, 'link': doc_info['link'], 'title': doc_info['title'], 'latex':info['latex'], 'summary':doc_info['summary'], 'embedding':info['embedding'], 'timestamp':info['timestamp']}
    print(len(body['embedding']))