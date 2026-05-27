import huggingface_hub

def get_gguf_file(repo_id, pattern="q4_k_m.gguf"):
    info = huggingface_hub.model_info(repo_id)
    for sibling in info.siblings:
        if pattern.lower() in sibling.rfilename.lower():
            return sibling.rfilename
    return None

print("File:", get_gguf_file("bartowski/Phi-3.5-mini-instruct-GGUF"))
