def search_similar_chunks(supabase, embedding_vector, match_count=5):
    response = supabase.rpc(
        "match_wiki",  # имя функции в Supabase
        {
            "query_embedding": embedding_vector.tolist(),
            "match_count": match_count
        }
    ).execute()

    if hasattr(response, "data"):
        return response.data
    else:
        print("Ошибка при запросе к базе данных:", response)
        return []