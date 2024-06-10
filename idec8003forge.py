import nest_asyncio
nest_asyncio.apply()

# Configure LlamaIndex
from llama_index.core import Settings
from llama_index.embeddings.llamafile import LlamafileEmbedding
from llama_index.llms.llamafile import Llamafile
from llama_index.core.node_parser import SentenceSplitter

Settings.embed_model = LlamafileEmbedding(base_url="http://localhost:8080")

Settings.llm = Llamafile(
	base_url="http://localhost:8080",
	temperature=0,
	seed=0
)

# Also set up a sentence splitter to ensure texts are broken into semantically-meaningful chunks (sentences) that don't take up the model's entire
# context window (2048 tokens). Since these chunks will be added to LLM prompts as part of the RAG process, we want to leave plenty of space for both
# the system prompt and the user's actual question.
Settings.transformations = [
	SentenceSplitter(
    	chunk_size=256,
    	chunk_overlap=5
	)
]

# Load local data
from llama_index.core import SimpleDirectoryReader
local_doc_reader = SimpleDirectoryReader(input_dir='../Data')
docs = local_doc_reader.load_data(show_progress=True)

# We'll load some Wikipedia pages as well
from llama_index.readers.web import SimpleWebPageReader
urls = [
	'https://en.wikipedia.org/wiki/Economic_development',
	'https://en.wikipedia.org/wiki/Policy',
]
web_reader = SimpleWebPageReader(html_to_text=True)
docs.extend(web_reader.load_data(urls))

# Build the index
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
	docs,
	show_progress=True,
)

# Save the index
index.storage_context.persist(persist_dir="../Storage")

query_engine = index.as_query_engine()
with open("./output.txt", "w") as f:
	f.writelines(query_engine.query("What forms does financial repression take in developing countries?"))



