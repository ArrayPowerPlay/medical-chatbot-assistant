from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class ContextualChunker:
    """This class can take input as a article, chunk it into many smaller chunks and add context to each
    chunk based on the context of the full article (title + abstract)"""
    def __init__(self, api_key: str, chunk_size: int = 800, chunk_overlap: int = 200):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0   # Greedy search
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def enrich_article(self, article: Dict) -> List[Dict]:
        """Add context into chunks of a article to enhance vector search"""
        title = article.get("title", "")
        abstract = article.get("abstractText", "")
        full_abstract = title + abstract
        chunks = self.splitter.split_text(full_abstract)

        # Process chunks in batches to save API costs (sending API only once per batche)
        batch_size = 20

        enrich_results = []
        parser = JsonOutputParser()
        formatted_instructions = parser.get_format_instructions()    # Extract JSON in LLM's response
        
        batch_prompt = ChatPromptTemplate.from_template("""
        You are a medical research assistant. Given the full abstract of a paper and some chunks from it, 
        give a list of short succinct context to situate EACH chunk within the overall document for the purposes of 
        improving search retrieval.
                                                        
        Return a JSON object with a single key "contexts", which contains a list of strings.
        Each string in the list should be the context for the corresponding chunk.
        The list of contexts must have the same number of elements as the list of chunks.
        
        Abstract: {abstract}
        
        Chunks to process:
        {chunks_list}

        {formatted_instructions}
        """)

        chain = batch_prompt | self.llm | parser

        for i in range(0, len(chunks), batch_size):
            current_batch = chunks[i: i + batch_size]
            chunks_formatted = "\n".join([f"Chunk {idx}: {c}" for idx, c in enumerate(current_batch)])
            
            contexts = chain.invoke({
                "abstract": full_abstract,
                "chunks_list": chunks_formatted,
                "formatted_instructions": formatted_instructions
            })

            context_list = contexts.get("contexts", [])
            for idx, chunk_text in enumerate(current_batch):
                ctx = context_list[idx] if idx < len(context_list) else "Medical context"
                enrich_results.append({
                    "text": f"Context: {ctx}\nChunk: {chunk_text}",
                    "pmid": article.get("pmid")
                })

        return enrich_results