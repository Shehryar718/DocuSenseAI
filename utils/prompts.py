generation_prompt = '''
Generate a detailed and concise description of the following file content: text(s), image(s), PDF(s), CSV(s), Excel(s).
The description should capture the essence of the content to assist in later ranking the relevance of the documents against a query string using cosine similarity.
Additionally, provide relevant tags that range from generic to specific.
For images, include a thorough description of the visual content, highlighting key elements, and any notable features. If the content is not clear, make a best guess based on visual elements.
Ensure the output follows this format:
Description: [Detailed description]
Tags: [comma-separated tags]
Avoid responses indicating inability to describe the content. Always provide a meaningful description.
'''

retrieval_prompt = """
Using the content from the provided document(s), please deliver the most accurate answer to my question.
If a document is relevant, include its path at the end in the format: 'Path: [path]'. 
If no document is relevant, say that there is no relevant document and OMIT the path.
"""